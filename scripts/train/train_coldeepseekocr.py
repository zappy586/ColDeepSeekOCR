"""
Fine-tune ColDeepSeekOCR on a subset of vidore/colpali_train_set.

Logging (via HF Trainer + W&B):
  - train loss      : every 100 train steps  (logging_steps=100)
  - val loss        : every 100 train steps  (eval_steps=100)
  - val recall@1    : every 500 train steps  (custom RetrievalAccuracyCallback)

Auth:
  Set WANDB_API_KEY in your environment BEFORE running. The script reads it from
  the env var; we never persist it. Example:
      export WANDB_API_KEY=wandb_v1_...
      python scripts/train/train_coldeepseekocr.py [--dry-run]

CLI:
  --dry-run      : tiny subset, no W&B, fast smoke-test
  --output-dir   : checkpoint dir (default ./models/coldeepseekocr_run)
  --batch-size   : per-device train batch size (default 1)
  --grad-accum   : gradient accumulation steps (default 4)
  --epochs       : number of epochs (default 2)
  --lr           : learning rate (default 2e-4)
  --max-train    : cap on training samples (default 9500)
  --max-eval     : cap on val samples (default 250)
  --max-test     : cap on test samples (default 250, currently unused)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Ensure colpali is importable when this script is run directly.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from datasets import Dataset as HFDataset, Features, Image as HFImage, Value, load_dataset, load_from_disk  # noqa: E402
from peft import LoraConfig, prepare_model_for_kbit_training                 # noqa: E402
from transformers import (                                                   # noqa: E402
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)

from colpali_engine.data.dataset import ColPaliEngineDataset                 # noqa: E402
from colpali_engine.loss.late_interaction_losses import ColbertLoss          # noqa: E402
from colpali_engine.models.deepseek_ocr import (                             # noqa: E402
    ColDeepSeekOCR,
    ColDeepSeekOCRProcessor,
)
from colpali_engine.trainer.colmodel_training import (                       # noqa: E402
    ColModelTraining,
    ColModelTrainingConfig,
)

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
DATASET_NAME = "vidore/colpali_train_set"


# ---------------------------------------------------------------------------
# Custom callback: val retrieval Recall@1 every N train steps
# ---------------------------------------------------------------------------
class RetrievalAccuracyCallback(TrainerCallback):
    """Encode the entire val set, compute MaxSim scores for every (query, doc)
    pair, and log Recall@1 to W&B / console every `eval_every` train steps."""

    def __init__(self, eval_dataset, processor, eval_every: int = 500, batch_size: int = 1):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.eval_every = eval_every
        self.batch_size = batch_size

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return control
        if model is None:
            return control

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        # Pull (query, image) pairs from the wrapped dataset.
        queries: List[str] = []
        images = []
        for i in range(len(self.eval_dataset)):
            sample = self.eval_dataset[i]
            queries.append(sample[self.eval_dataset.QUERY_KEY])
            pos = sample[self.eval_dataset.POS_TARGET_KEY]
            images.append(pos[0] if isinstance(pos, list) else pos)

        try:
            with torch.no_grad():
                # --- doc embeddings (one image at a time to keep memory bounded) ---
                d_embs = []
                for i in range(0, len(images), self.batch_size):
                    chunk = images[i : i + self.batch_size]
                    d_in = self.processor.process_images(chunk)
                    d_in_gpu = {
                        "input_ids":           d_in["input_ids"].to(device),
                        "attention_mask":      d_in["attention_mask"].to(device),
                        "images":              [(c.to(device), g.to(device)) for c, g in d_in["images"]],
                        "images_seq_mask":     d_in["images_seq_mask"].to(device),
                        "images_spatial_crop": d_in["images_spatial_crop"].to(device),
                    }
                    out = model(**d_in_gpu)  # (b, seq, dim)
                    for j in range(out.size(0)):
                        mask = d_in["attention_mask"][j].bool()
                        d_embs.append(out[j][mask].detach().float().cpu())

                # --- query embeddings ---
                q_embs = []
                for i in range(0, len(queries), self.batch_size):
                    chunk = queries[i : i + self.batch_size]
                    q_in = self.processor.process_texts(chunk)
                    out = model(
                        input_ids=q_in["input_ids"].to(device),
                        attention_mask=q_in["attention_mask"].to(device),
                    )
                    for j in range(out.size(0)):
                        mask = q_in["attention_mask"][j].bool()
                        q_embs.append(out[j][mask].detach().float().cpu())

                # --- MaxSim scoring ---
                hits = 0
                for qi, q in enumerate(q_embs):
                    best_score = -float("inf")
                    best_idx = -1
                    for di, d in enumerate(d_embs):
                        # MaxSim: sum over query tokens of max similarity to any doc token
                        score = (q @ d.T).max(dim=1).values.sum().item()
                        if score > best_score:
                            best_score = score
                            best_idx = di
                    if best_idx == qi:
                        hits += 1
                recall_at_1 = hits / len(q_embs)
        finally:
            if was_training:
                model.train()

        # Log to whatever the trainer is reporting to (W&B, console, etc.)
        log_payload = {"val/recall@1": recall_at_1, "step": state.global_step}
        print(f"[step {state.global_step}] val/recall@1 = {recall_at_1:.4f}")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(log_payload, step=state.global_step)
        except Exception:
            pass
        return control


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-dir", default="./models/coldeepseekocr_run")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-train", type=int, default=9500)
    p.add_argument("--max-eval", type=int, default=250)
    p.add_argument("--max-test", type=int, default=250)
    p.add_argument("--logging-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--accuracy-eval-steps", type=int, default=500)
    p.add_argument("--no-lora", action="store_true", help="Train full model instead of LoRA")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default 16)")
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (scaling). Defaults to 2 * lora_r if unset.",
    )
    p.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout (default 0.1)")
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Linear LR warmup steps (default 50; 0 disables warmup, dry-run forces 0).",
    )
    p.add_argument(
        "--head-only",
        action="store_true",
        help="Freeze the entire backbone and train only custom_text_proj. "
             "Implies --no-lora. Lets you use much larger batches since no "
             "backbone gradients/optimizer states are stored.",
    )
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the backbone in 4-bit (NF4 + double quant) via bitsandbytes. "
             "Pairs naturally with LoRA (QLoRA) — backbone stays frozen at low "
             "precision, LoRA adapters train in bf16. Frees ~5GB VRAM.",
    )
    args = p.parse_args()
    if args.head_only:
        args.no_lora = True

    if args.dry_run:
        # Tiny smoke-test config
        args.max_train, args.max_eval, args.max_test = 8, 4, 4
        args.epochs = 1
        args.logging_steps, args.eval_steps, args.accuracy_eval_steps = 2, 2, 4
        args.output_dir = "/tmp/coldeepseekocr_dryrun"

    # ------- W&B setup -------
    if not args.dry_run:
        if "WANDB_API_KEY" not in os.environ:
            raise RuntimeError(
                "WANDB_API_KEY env var not set. Export it before running:\n"
                "    export WANDB_API_KEY=<your-key>"
            )
        os.environ.setdefault("WANDB_PROJECT", "coldeepseekocr")
        report_to = "wandb"
    else:
        report_to = "none"
        os.environ["WANDB_DISABLED"] = "true"

    # ------- Dataset -------
    # Stream the desired subset *once* and save it to disk as an Arrow shard.
    # Subsequent runs reuse the on-disk shard. Once on disk, datasets memory-maps
    # the Arrow file so each (query, image) is only decoded when its row is
    # accessed — RAM stays bounded even for 10k images.
    total = args.max_train + args.max_eval + args.max_test
    subset_path = Path(f"./data_dir/vidore_colpali_train_set_subset_{total}")
    if subset_path.exists():
        print(f"Loading cached subset from {subset_path}")
        raw = load_from_disk(str(subset_path))
    else:
        print(f"Streaming {DATASET_NAME} → {subset_path} (taking {total} samples)...")
        stream = load_dataset(DATASET_NAME, split="train", streaming=True)

        import io

        def _gen():
            for i, ex in enumerate(stream):
                if i >= total:
                    break
                if (i + 1) % 500 == 0:
                    print(f"  streamed {i + 1}/{total}")
                # Encode the image as PNG bytes BEFORE yielding. This keeps the
                # decoded PIL out of the writer's batch buffer — a buffered batch
                # of ~1MB-each PNGs uses an order of magnitude less RAM than
                # decoded 1024x1024 RGB tensors.
                img = ex["image"]
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                yield {"query": ex["query"], "image": {"bytes": buf.getvalue()}}

        features = Features({
            "query": Value("string"),
            "image": HFImage(decode=True),  # auto-decodes PNG bytes back to PIL on access
        })
        # writer_batch_size=50 → at most 50 PNG-encoded images (≤100MB total) buffered
        # before the writer flushes to disk. Keeps RAM bounded.
        raw = HFDataset.from_generator(_gen, features=features, writer_batch_size=50)
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save_to_disk(str(subset_path))
        # Reload from disk so subsequent reads are memory-mapped (lazy).
        raw = load_from_disk(str(subset_path))

    raw = raw.shuffle(seed=42)
    train_raw = raw.select(range(args.max_train))
    eval_raw  = raw.select(range(args.max_train, args.max_train + args.max_eval))
    # test_raw  = raw.select(range(args.max_train + args.max_eval, total))

    train_ds = ColPaliEngineDataset(train_raw, pos_target_column_name="image")
    eval_ds  = ColPaliEngineDataset(eval_raw,  pos_target_column_name="image")
    print(f"  train={len(train_ds)} val={len(eval_ds)}")

    # ------- Model + Processor -------
    print("Loading model + processor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    processor = ColDeepSeekOCRProcessor(tokenizer)

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_safetensors=True,
        ignore_mismatched_sizes=True,
    )
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # bitsandbytes wants a device_map — let HF auto-place it on the visible GPU.
        load_kwargs["device_map"] = "auto"
        print("Loading backbone in 4-bit (NF4 + double quant)...")

    model = ColDeepSeekOCR.from_pretrained(MODEL_NAME, **load_kwargs)

    if args.load_in_4bit:
        # Cast layer norms / projection layer to fp32 for stable LoRA training,
        # and enable gradient checkpointing-compatible hooks on quantized layers.
        # (No-op for layers that aren't quantized.)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )

    # ------- Head-only mode: freeze everything except custom_text_proj -------
    if args.head_only:
        n_frozen = 0
        n_train = 0
        for name, p_ in model.named_parameters():
            if "custom_text_proj" in name:
                p_.requires_grad = True
                n_train += p_.numel()
            else:
                p_.requires_grad = False
                n_frozen += p_.numel()
        print(f"Head-only mode: trainable={n_train:,} frozen={n_frozen:,} "
              f"(trainable%={100 * n_train / (n_train + n_frozen):.4f})")

    # ------- LoRA (default) -------
    peft_config = None
    if not args.no_lora:
        lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r
        print(f"LoRA config: r={args.lora_r}, alpha={lora_alpha}, dropout={args.lora_dropout}")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            # Target the LLM proj layers + our custom_text_proj.
            target_modules=r"(.*(model\.layers).*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*$|.*(custom_text_proj).*$)",
        )

    # ------- TrainingArguments -------
    tr_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=0 if args.dry_run else args.warmup_steps,
        bf16=True,
        gradient_checkpointing=False,  # DeepseekOCRModel doesn't support it cleanly
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=max(args.eval_steps * 5, 500),
        save_total_limit=2,
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        run_name=os.environ.get("WANDB_RUN_NAME", "coldeepseekocr_run"),
    )

    cfg = ColModelTrainingConfig(
        model=model,
        processor=processor,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tr_args=tr_args,
        output_dir=args.output_dir,
        loss_func=ColbertLoss(),
        peft_config=peft_config,
    )

    training_app = ColModelTraining(cfg)

    # Inject the retrieval-accuracy callback.
    accuracy_cb = RetrievalAccuracyCallback(
        eval_dataset=eval_ds,
        processor=processor,
        eval_every=args.accuracy_eval_steps,
        batch_size=args.batch_size,
    )

    # ColModelTraining.train() builds its own trainer internally; we wrap that.
    from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
    trainer = ContrastiveTrainer(
        model=training_app.model,
        train_dataset=training_app.train_dataset,
        eval_dataset=training_app.eval_dataset,
        args=tr_args,
        data_collator=training_app.collator,
        loss_func=cfg.loss_func,
        is_vision_model=True,
    )
    trainer.args.remove_unused_columns = False
    trainer.add_callback(accuracy_cb)

    print("Starting training...")
    trainer.train()
    if not args.dry_run:
        training_app.save()
    print("Done.")


if __name__ == "__main__":
    main()
