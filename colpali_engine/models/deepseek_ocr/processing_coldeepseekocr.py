import math
from typing import ClassVar, List, Optional, Set, Tuple, Union

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import BatchFeature

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


# ---------------------------------------------------------------------------
# Preprocessing utilities
#
# These are pure functions extracted from DeepseekOCRForCausalLM.infer() so
# that the processor can be imported independently of the remote-code files
# (deepencoder.py / conversation.py) that DeepseekOCRForCausalLM requires.
# ---------------------------------------------------------------------------

def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 2,
    max_num: int = 9,
    image_size: int = 640,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    target_ratios: List[Tuple[int, int]] = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    best = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    target_w, target_h = image_size * best[0], image_size * best[1]

    resized = image.resize((target_w, target_h))
    patches: List[Image.Image] = []
    cols = target_w // image_size
    for i in range(best[0] * best[1]):
        col, row = i % cols, i // cols
        box = (col * image_size, row * image_size, (col + 1) * image_size, (row + 1) * image_size)
        patches.append(resized.crop(box))

    return patches, best


def _tokenize(tokenizer, text: str, bos: bool = True) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ([0] + ids) if bos else ids   # BOS token id = 0 for the DeepSeek tokenizer


class _ImageTransform:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class ColDeepSeekOCRProcessor(BaseVisualRetrieverProcessor):
    """
    Processor for ColDeepSeekOCR.

    Replicates the image preprocessing pipeline of DeepseekOCRForCausalLM.infer() and exposes
    it through the standard ColPali process_images / process_texts interface.

    Args:
        tokenizer: DeepSeek-OCR tokenizer (load with trust_remote_code=True).
        base_size: Height/width to which the global view is padded. Default 1024.
        image_size: Tile size used for dynamic aspect-ratio cropping. Default 640.
        crop_mode: Use dynamic tiling (True) or single-resize (False). Default True.
    """

    image_token: ClassVar[str] = "<image>"
    image_token_id: ClassVar[int] = 128815
    visual_prompt_prefix: ClassVar[str] = "<image>\nDescribe the image."
    # Query-side fields consumed by colpali_engine.collators.VisualRetrieverCollator:
    #   final_query = query_prefix + <query text> + query_augmentation_token * N
    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = "<｜end▁of▁sentence｜>"

    def __init__(
        self,
        tokenizer,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
    ):
        self.tokenizer = tokenizer
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode

        patch_size = 16
        downsample = 4
        self.num_queries_base = math.ceil((base_size // patch_size) / downsample)   # 16 @ 1024
        self.num_queries = math.ceil((image_size // patch_size) / downsample)       # 10 @ 640

        self._transform = _ImageTransform()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_image(
        self, image: Image.Image
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor, List[int]]:
        """
        Preprocess a single PIL image.

        Returns:
            img_token_ids : list of image_token_id values to insert into input_ids.
            global_tensor : (1, 3, base_size, base_size) bfloat16 global view.
            crop_tensor   : (num_crops, 3, image_size, image_size) bfloat16 patch tiles,
                            or a (1, 3, base_size, base_size) zero tensor when there are none.
            spatial_crop  : [width_crops, height_crops].
        """
        image = image.convert("RGB")
        pad_color = tuple(int(v * 255) for v in self._transform.mean)

        if self.crop_mode:
            if image.size[0] <= self.image_size and image.size[1] <= self.image_size:
                crop_ratio: Tuple[int, int] = (1, 1)
                raw_patches: List[Image.Image] = []
            else:
                raw_patches, crop_ratio = _dynamic_preprocess(image, image_size=self.image_size)

            w_crops, h_crops = crop_ratio
            global_view = ImageOps.pad(image, (self.base_size, self.base_size), color=pad_color)
            global_tensor = self._transform(global_view).to(torch.bfloat16).unsqueeze(0)

            crop_tensors = [self._transform(p).to(torch.bfloat16) for p in raw_patches]
            crop_tensor = (
                torch.stack(crop_tensors) if crop_tensors
                else torch.zeros(1, 3, self.base_size, self.base_size, dtype=torch.bfloat16)
            )

            # Image-token sequence matching the model's expected layout
            nqb, nq = self.num_queries_base, self.num_queries
            img_token_ids = ([self.image_token_id] * nqb + [self.image_token_id]) * nqb
            img_token_ids += [self.image_token_id]
            if w_crops > 1 or h_crops > 1:
                img_token_ids += (
                    [self.image_token_id] * (nq * w_crops) + [self.image_token_id]
                ) * (nq * h_crops)

        else:
            w_crops, h_crops = 1, 1
            global_view = ImageOps.pad(image, (self.image_size, self.image_size), color=pad_color)
            global_tensor = self._transform(global_view).to(torch.bfloat16).unsqueeze(0)
            crop_tensor = torch.zeros(1, 3, self.base_size, self.base_size, dtype=torch.bfloat16)

            nq = self.num_queries
            img_token_ids = ([self.image_token_id] * nq + [self.image_token_id]) * nq
            img_token_ids += [self.image_token_id]

        return img_token_ids, global_tensor, crop_tensor, [w_crops, h_crops]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_images(self, images: List[Image.Image]) -> BatchFeature:
        """
        Process a batch of document images into model inputs.

        Returns a BatchFeature with:
            input_ids          (batch, max_seq)  — right-padded with 0
            attention_mask     (batch, max_seq)  — 1 real / 0 pad
            images             list[(crop_tensor, global_tensor)] one tuple per sample
            images_seq_mask    (batch, max_seq)  — True at image-token positions
            images_spatial_crop (batch, 2)       — [width_crops, height_crops] per sample
        """
        prefix_text, suffix_text = self.visual_prompt_prefix.split(self.image_token, maxsplit=1)
        prefix_ids = _tokenize(self.tokenizer, prefix_text, bos=False)
        suffix_ids = _tokenize(self.tokenizer, suffix_text, bos=False)

        all_ids: List[torch.Tensor] = []
        all_seq_masks: List[torch.Tensor] = []
        images_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        spatial_crops: List[List[int]] = []

        for image in images:
            img_ids, global_t, crop_t, spatial = self._encode_image(image)

            # [BOS=0] prefix  <image tokens>  suffix
            ids = [0] + prefix_ids + img_ids + suffix_ids
            seq_mask = (
                [False] * (1 + len(prefix_ids))
                + [True] * len(img_ids)
                + [False] * len(suffix_ids)
            )

            all_ids.append(torch.tensor(ids, dtype=torch.long))
            all_seq_masks.append(torch.tensor(seq_mask, dtype=torch.bool))
            images_list.append((crop_t, global_t))
            spatial_crops.append(spatial)

        # Right-pad all sequences to the same length
        max_len = max(t.shape[0] for t in all_ids)
        n = len(images)
        padded_ids = torch.zeros(n, max_len, dtype=torch.long)
        padded_attn = torch.zeros(n, max_len, dtype=torch.long)
        padded_seq = torch.zeros(n, max_len, dtype=torch.bool)

        for i, (ids, seq_mask) in enumerate(zip(all_ids, all_seq_masks)):
            L = ids.shape[0]
            padded_ids[i, :L] = ids
            padded_attn[i, :L] = 1
            padded_seq[i, :L] = seq_mask

        return BatchFeature({
            "input_ids": padded_ids,
            "attention_mask": padded_attn,
            "images": images_list,
            "images_seq_mask": padded_seq,
            "images_spatial_crop": torch.tensor(spatial_crops, dtype=torch.long),
        })

    def process_texts(self, texts: List[str]) -> BatchFeature:
        """
        Process a batch of query strings into model inputs (no images).

        Returns a BatchFeature with input_ids and attention_mask only.
        ColDeepSeekOCR.forward() will automatically inject dummy zero images so that
        the model's image-injection branch is skipped.
        """
        tokenized = [_tokenize(self.tokenizer, t, bos=True) for t in texts]
        max_len = max(len(t) for t in tokenized)
        n = len(texts)

        padded_ids = torch.zeros(n, max_len, dtype=torch.long)
        padded_attn = torch.zeros(n, max_len, dtype=torch.long)

        for i, t in enumerate(tokenized):
            L = len(t)
            padded_ids[i, :L] = torch.tensor(t, dtype=torch.long)
            padded_attn[i, :L] = 1

        return BatchFeature({"input_ids": padded_ids, "attention_mask": padded_attn})

    def score(
        self,
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """MaxSim (ColBERT-style) late-interaction score between queries and passages."""
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Patch grid for the global view (square `base_size` image, projected down
        to `num_queries_base` × `num_queries_base` query tokens).
        """
        return self.num_queries_base, self.num_queries_base
