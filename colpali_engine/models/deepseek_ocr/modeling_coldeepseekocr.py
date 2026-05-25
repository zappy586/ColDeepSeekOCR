import os
import shutil
from pathlib import Path
from typing import ClassVar, Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Auto-patch the hub's modeling_deepseekv2.py
#
# The deepseek-ai/DeepSeek-OCR repo on the Hub ships a modeling_deepseekv2.py
# that imports `LlamaAttention`, `LlamaFlashAttention2`, `is_flash_attn_2_available`,
# `is_torch_fx_available`, etc. from transformers. These symbols were removed in
# recent transformers versions, so the import fails. We bundle a patched copy
# of the file alongside this module and inject it into the HF dynamic-module
# cache before the first `get_class_from_dynamic_module` call.
# ---------------------------------------------------------------------------
_REPO_ID = "deepseek-ai/DeepSeek-OCR"
_PATCH_MARKER = "class DeepseekMHAAttention"  # presence == patched
_PATCHED_FILE = Path(__file__).parent / "_patched_modeling_deepseekv2.py"


def _ensure_remote_code_patched() -> None:
    """Download deepseek-ai/DeepSeek-OCR's remote code (if needed) and patch
    `modeling_deepseekv2.py` in the HF module cache so it loads on modern
    transformers versions. Idempotent."""
    try:
        from huggingface_hub import snapshot_download
        from transformers.dynamic_module_utils import HF_MODULES_CACHE
    except Exception:
        return  # If hub/transformers utilities aren't available, fall through.

    # Trigger a download of just the .py files. snapshot_download is a no-op if
    # they're already present.
    try:
        snapshot_download(
            repo_id=_REPO_ID,
            allow_patterns=["*.py", "config.json", "tokenizer*"],
        )
    except Exception:
        # Offline / no network: assume the user already has the cache populated.
        pass

    # transformers copies the .py files from the hub-snapshot dir into
    # HF_MODULES_CACHE/transformers_modules/<org>/<repo>/<commit>/ the first
    # time `get_class_from_dynamic_module` runs. Before that, the directory
    # doesn't exist. Trigger that copy by importing the (broken) module via
    # `get_cached_module_file`, which only copies — it doesn't import.
    try:
        from transformers.dynamic_module_utils import get_cached_module_file
        get_cached_module_file(_REPO_ID, "modeling_deepseekv2.py")
    except Exception:
        # The import inside the cached file fails, but the file has been
        # copied into HF_MODULES_CACHE by now, which is all we need.
        pass

    # Find the cached modeling_deepseekv2.py and patch it if needed.
    sanitized = _REPO_ID.replace("-", "_hyphen_").replace("/", "/")
    module_root = Path(HF_MODULES_CACHE) / "transformers_modules" / sanitized
    if not module_root.exists():
        # Fallback: also try the un-sanitized layout
        module_root = Path(HF_MODULES_CACHE) / "transformers_modules" / _REPO_ID
    if not module_root.exists():
        return

    for cached_file in module_root.rglob("modeling_deepseekv2.py"):
        try:
            text = cached_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if _PATCH_MARKER in text:
            continue  # already patched
        if not _PATCHED_FILE.exists():
            continue  # bundled copy missing — skip silently
        shutil.copyfile(_PATCHED_FILE, cached_file)
        # Bust any cached bytecode so the next import picks up the new file.
        pycache = cached_file.parent / "__pycache__"
        if pycache.exists():
            shutil.rmtree(pycache, ignore_errors=True)

    # ---- silence DeepSeek's per-forward debug prints in modeling_deepseekocr.py ----
    # DeepSeek's hub code prints vision-feature shapes ("BASE:", "PATCHES:", "=====")
    # on every forward pass. They spam the training log. Comment them out.
    _DEBUG_PRINT_LINES = (
        "print('=====================')",
        "print('BASE: ', global_features.shape)",
        "print('PATCHES: ', local_features.shape)",
        "print('NO PATCHES')",
    )
    _SILENCED_MARKER = "# [coldeepseek] debug prints silenced"
    for cached_file in module_root.rglob("modeling_deepseekocr.py"):
        try:
            text = cached_file.read_text(encoding="utf-8")
        except Exception:
            continue
        if _SILENCED_MARKER in text:
            continue  # already silenced
        new_text = text
        for line in _DEBUG_PRINT_LINES:
            new_text = new_text.replace(line, f"pass  # {line}")
        if new_text == text:
            continue  # nothing matched (file structure changed upstream)
        new_text += f"\n{_SILENCED_MARKER}\n"
        cached_file.write_text(new_text, encoding="utf-8")
        pycache = cached_file.parent / "__pycache__"
        if pycache.exists():
            shutil.rmtree(pycache, ignore_errors=True)


_ensure_remote_code_patched()


# Load DeepseekOCRForCausalLM from the HuggingFace module cache (populated when
# deepseek-ai/DeepSeek-OCR is first fetched with trust_remote_code=True).
# This avoids importing the local transformers fork, which conflicts with the
# hub's remote code on newer transformers versions.
try:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    DeepseekOCRForCausalLM = get_class_from_dynamic_module(
        "modeling_deepseekocr.DeepseekOCRForCausalLM",
        _REPO_ID,
    )
    DeepseekOCRConfig = get_class_from_dynamic_module(
        "modeling_deepseekocr.DeepseekOCRConfig",
        _REPO_ID,
    )
except Exception:
    from transformers import PreTrainedModel, PretrainedConfig
    DeepseekOCRForCausalLM = PreTrainedModel  # type: ignore[assignment,misc]
    DeepseekOCRConfig = PretrainedConfig       # type: ignore[assignment,misc]


class ColDeepSeekOCR(DeepseekOCRForCausalLM):
    """
    ColDeepSeekOCR: DeepSeek-OCR adapted as a ColPali-style multi-vector document embedding model.

    Adds a linear projection layer and L2 normalisation on top of DeepseekOCRForCausalLM to
    produce per-token embeddings suitable for late-interaction retrieval (MaxSim / ColBERT).

    The model hub checkpoint (deepseek-ai/DeepSeek-OCR) must have been fetched at least once
    with trust_remote_code=True so that the remote code is present in the HF module cache.

    Args:
        config: Model configuration (DeepseekOCRConfig).
        mask_non_image_embeddings: If True, zero out every token position that is not an image
            token when images are provided. Defaults to False.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"
    image_token_id: ClassVar[int] = 128815
    default_base_size: ClassVar[int] = 1024

    def __init__(self, config, mask_non_image_embeddings: bool = False):
        if hasattr(config, "language_config") and isinstance(config.language_config, dict):
            for k, v in config.language_config.items():
                if not hasattr(config, k):
                    setattr(config, k, v)
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            DeepseekV2Config = get_class_from_dynamic_module(
                "configuration_deepseek_v2.DeepseekV2Config",
                "deepseek-ai/DeepSeek-OCR",
            )
            defaults = DeepseekV2Config()
            for k, v in defaults.__dict__.items():
                if not hasattr(config, k):
                    setattr(config, k, v)
        except Exception:
            pass
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = None
        super().__init__(config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(config.hidden_size, self.dim)
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        # If the backbone was loaded in 4-bit (QLoRA path), bitsandbytes wraps
        # EVERY nn.Linear in the model — including our randomly-initialised
        # `custom_text_proj`. But because that layer is not in the checkpoint,
        # its weight never went through the quantisation pipeline, leaving a
        # `Linear4bit` shell around a plain `Parameter`. PEFT chokes on that.
        # Replace it with a fresh full-precision nn.Linear (we want to train it
        # in bf16 anyway).
        try:
            import bitsandbytes as bnb
            if isinstance(model.custom_text_proj, bnb.nn.Linear4bit):
                in_f  = model.custom_text_proj.in_features
                out_f = model.custom_text_proj.out_features
                dev   = next(p.device for p in model.parameters())
                model.custom_text_proj = nn.Linear(in_f, out_f).to(dev, dtype=torch.bfloat16)
        except Exception:
            pass

        with torch.no_grad():
            # CLIPVisionEmbeddings.position_ids — should be arange(num_positions).
            emb = model.model.vision_model.embeddings
            emb.position_ids.copy_(
                torch.arange(emb.num_positions, device=emb.position_ids.device).expand((1, -1))
            )
            # DeepseekV2RotaryEmbedding.{inv_freq,cos_cached,sin_cached} on every
            # attention layer — derived from (dim, base) and need re-derivation.
            for module in model.modules():
                if hasattr(module, "inv_freq") and hasattr(module, "dim") and hasattr(module, "base"):
                    dev = module.inv_freq.device
                    inv_freq = 1.0 / (
                        module.base ** (torch.arange(0, module.dim, 2, device=dev).float() / module.dim)
                    )
                    module.inv_freq.copy_(inv_freq.to(module.inv_freq.dtype))
                    if hasattr(module, "cos_cached") and hasattr(module, "sin_cached"):
                        seq = module.max_position_embeddings
                        t = torch.arange(seq, device=dev, dtype=inv_freq.dtype)
                        freqs = torch.outer(t, inv_freq)
                        emb_cache = torch.cat((freqs, freqs), dim=-1)
                        module.cos_cached.copy_(emb_cache.cos().to(module.cos_cached.dtype))
                        module.sin_cached.copy_(emb_cache.sin().to(module.sin_cached.dtype))
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images=None,
        images_seq_mask: Optional[torch.BoolTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Returns per-token L2-normalised embeddings of shape (batch, seq, dim=128).

        Args:
            input_ids: Token IDs.  Image placeholder positions use id=128815.
            attention_mask: 1 for real tokens, 0 for padding.
            images: List of (crop_tensor, global_view_tensor) tuples, one per batch item.
                    crop_tensor  shape: (num_crops, 3, image_size, image_size)
                    global_view  shape: (1, 3, base_size, base_size)
                    Pass None for text-only (query) inputs.
            images_seq_mask: Bool tensor (batch, seq) — True at image-token positions.
            images_spatial_crop: Long tensor (batch, 2) — [width_crops, height_crops] per image.
        """
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("labels", None)
        kwargs.pop("use_cache", None)

        has_images = images is not None

        if input_ids is not None:
            batch_size, device = input_ids.shape[0], input_ids.device
        else:
            batch_size, device = inputs_embeds.shape[0], inputs_embeds.device

        if not has_images:
            # DeepseekOCRModel.forward() unconditionally accesses images[0][1], so we must
            # always pass something.  All-zero tensors cause the image-injection branch to be
            # skipped via its own guard: `torch.sum(images[0][1]) != 0`.
            dummy = torch.zeros(
                1, 3, self.default_base_size, self.default_base_size,
                dtype=torch.bfloat16, device=device,
            )
            images = [(dummy, dummy)] * batch_size
            images_seq_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            images_spatial_crop = torch.zeros(batch_size, 2, dtype=torch.long, device=device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

        last_hidden_states = outputs.last_hidden_state            # (batch, seq, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)          # (batch, seq, dim)
        proj = proj / proj.norm(dim=-1, keepdim=True)             # L2 normalise
        proj = proj * attention_mask.unsqueeze(-1).float()        # zero padding positions

        if has_images and self.mask_non_image_embeddings:
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1)
            proj = proj * image_mask

        return proj
