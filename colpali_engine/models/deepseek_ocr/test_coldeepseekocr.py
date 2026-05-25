"""
Quick functional test for ColDeepSeekOCR + ColDeepSeekOCRProcessor.

Run from the project root:
    cd /home/zeeshan/projects/project-1
    TRANSFORMERS_NO_FLASH_ATTENTION=1 CUDA_VISIBLE_DEVICES=0 \
        python colpali/colpali_engine/models/deepseek_ocr/test_coldeepseekocr.py
"""
import os, sys

# colpali package directory (3 levels up from deepseek_ocr/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

os.environ.setdefault("TRANSFORMERS_NO_FLASH_ATTENTION", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from PIL import Image
from transformers import AutoTokenizer

from colpali_engine.models.deepseek_ocr import ColDeepSeekOCR, ColDeepSeekOCRProcessor

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
SAMPLE_IMAGE = os.path.join(os.path.dirname(__file__), "sample_image.png")

# ---------------------------------------------------------------------------
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)

# ---------------------------------------------------------------------------
print("Loading ColDeepSeekOCR ...")
# custom_text_proj weights will be randomly initialised (not in the base checkpoint).
# In production you would fine-tune or save/load from a ColDeepSeekOCR checkpoint.
model = ColDeepSeekOCR.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    use_safetensors=True,
    ignore_mismatched_sizes=True,   # custom_text_proj is new
)
model = model.eval().cuda()
print(f"  base class : {type(model).__bases__[0].__name__}")
print(f"  dim        : {model.dim}")
print(f"  proj layer : {model.custom_text_proj}")

# ---------------------------------------------------------------------------
processor = ColDeepSeekOCRProcessor(tokenizer)

# ---------------------------------------------------------------------------
print("\n--- process_images ---")
image = Image.open(SAMPLE_IMAGE).convert("RGB")
doc_inputs = processor.process_images([image, image])   # batch of 2

print("  input_ids          :", doc_inputs["input_ids"].shape)
print("  attention_mask     :", doc_inputs["attention_mask"].shape)
print("  images_seq_mask    :", doc_inputs["images_seq_mask"].shape)
print("  images_spatial_crop:", doc_inputs["images_spatial_crop"])
print("  images (list len)  :", len(doc_inputs["images"]),
      " | crop shape:", doc_inputs["images"][0][0].shape,
      " | global shape:", doc_inputs["images"][0][1].shape)

# Move tensors to GPU; images list must be handled manually
doc_inputs_gpu = {
    "input_ids":           doc_inputs["input_ids"].cuda(),
    "attention_mask":      doc_inputs["attention_mask"].cuda(),
    "images":              [(c.cuda(), g.cuda()) for c, g in doc_inputs["images"]],
    "images_seq_mask":     doc_inputs["images_seq_mask"].cuda(),
    "images_spatial_crop": doc_inputs["images_spatial_crop"].cuda(),
}

with torch.no_grad():
    doc_embeddings = model(**doc_inputs_gpu)

print("  doc_embeddings shape:", doc_embeddings.shape)   # (2, seq, 128)
print("  norm (should be ~1) :", doc_embeddings[doc_inputs["attention_mask"].bool()].norm(dim=-1).mean().item())

# ---------------------------------------------------------------------------
print("\n--- process_texts ---")
queries = ["Who were the authors of the batch norm paper?", "What is depicted in this image?"]
text_inputs = processor.process_texts(queries)

print("  input_ids     :", text_inputs["input_ids"].shape)
print("  attention_mask:", text_inputs["attention_mask"].shape)

with torch.no_grad():
    query_embeddings = model(
        input_ids=text_inputs["input_ids"].cuda(),
        attention_mask=text_inputs["attention_mask"].cuda(),
    )

print("  query_embeddings shape:", query_embeddings.shape)   # (2, seq, 128)

# ---------------------------------------------------------------------------
print("\n--- MaxSim scores ---")
# Convert to list of per-sample tensors (strip padding)
q_list = [query_embeddings[i][text_inputs["attention_mask"][i].bool()] for i in range(len(queries))]
p_list = [doc_embeddings[i][doc_inputs["attention_mask"][i].bool()] for i in range(2)]

scores = processor.score(q_list, p_list)
print("  scores (2 queries x 2 docs):\n", scores)

print("\nAll checks passed.")
