#!/usr/bin/env python3
"""Fine-tune Mistral-7B-Instruct on MedDRA ontology SFT data using Unsloth QLoRA.

Usage (local test, small run):
    PYTHONPATH=. python experiments/train_ontology_lora.py \
        --train data/sft/ontology_train.jsonl \
        --val   data/sft/ontology_val.jsonl \
        --out   models/mistral-meddra-lora \
        --max-steps 100

Usage (full cluster run — called by SLURM script):
    PYTHONPATH=. python experiments/train_ontology_lora.py \
        --train data/sft/ontology_train.jsonl \
        --val   data/sft/ontology_val.jsonl \
        --out   models/mistral-meddra-lora
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
#  Hyper-parameters (override via CLI or env vars)
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_ID  = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MAX_SEQ   = 1024
DEFAULT_BATCH     = 4          # per-device; effective = batch * grad_accum
DEFAULT_GRAD_ACC  = 4          # effective batch = 16
DEFAULT_EPOCHS    = 3
DEFAULT_LR        = 2e-4
DEFAULT_LORA_R    = 16
DEFAULT_LORA_A    = 32
DEFAULT_LORA_DROP = 0.05

# Alpaca prompt template (must match build_ontology_sft.py output format)
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input. "
    "Write a response that completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

EOS_TOKEN = "</s>"  # Unsloth needs this appended


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _format_row(row: dict) -> str:
    return ALPACA_TEMPLATE.format(
        instruction=row.get("instruction", ""),
        input=row.get("input", ""),
        output=row.get("output", ""),
    ) + EOS_TOKEN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",       type=Path, required=True)
    ap.add_argument("--val",         type=Path, default=None)
    ap.add_argument("--out",         type=Path, required=True)
    ap.add_argument("--model-id",    default=os.environ.get("BASE_MODEL", DEFAULT_MODEL_ID))
    ap.add_argument("--max-seq",     type=int,  default=DEFAULT_MAX_SEQ)
    ap.add_argument("--batch",       type=int,  default=DEFAULT_BATCH)
    ap.add_argument("--grad-acc",    type=int,  default=DEFAULT_GRAD_ACC)
    ap.add_argument("--epochs",      type=int,  default=DEFAULT_EPOCHS)
    ap.add_argument("--max-steps",   type=int,  default=-1, help="Override epochs (debug).")
    ap.add_argument("--lr",          type=float, default=DEFAULT_LR)
    ap.add_argument("--lora-r",      type=int,  default=DEFAULT_LORA_R)
    ap.add_argument("--lora-alpha",  type=int,  default=DEFAULT_LORA_A)
    ap.add_argument("--lora-drop",   type=float, default=DEFAULT_LORA_DROP)
    ap.add_argument("--merge-weights", action="store_true",
                    help="Merge LoRA into base weights and save (bigger, standalone model).")
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    #  Imports (after arg parse so --help works without GPU)
    # ------------------------------------------------------------------ #
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise SystemExit(
            "unsloth not installed. Install order:\n"
            "  pip install unsloth_zoo\n"
            "  pip install 'unsloth[cu124]' --break-system-packages\n"
            f"Original error: {e}"
        ) from e

    import torch
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"[train_ontology_lora] CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[train_ontology_lora] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[train_ontology_lora] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------ #
    #  Load model with 4-bit QLoRA via Unsloth
    # ------------------------------------------------------------------ #
    print(f"[train_ontology_lora] Loading {args.model_id} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq,
        dtype=None,          # auto-detect (bf16 on L40S)
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_drop,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    # ------------------------------------------------------------------ #
    #  Data
    # ------------------------------------------------------------------ #
    train_rows = _load_jsonl(args.train)
    train_texts = [_format_row(r) for r in train_rows]
    train_ds = Dataset.from_dict({"text": train_texts})
    print(f"[train_ontology_lora] Train rows: {len(train_ds)}")

    eval_ds = None
    if args.val and args.val.is_file():
        val_rows = _load_jsonl(args.val)
        eval_texts = [_format_row(r) for r in val_rows]
        eval_ds = Dataset.from_dict({"text": eval_texts})
        print(f"[train_ontology_lora] Val rows:   {len(eval_ds)}")

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_ds else "no",
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq,
        args=training_args,
    )

    print("[train_ontology_lora] Starting training …")
    trainer.train()

    # ------------------------------------------------------------------ #
    #  Save
    # ------------------------------------------------------------------ #
    if args.merge_weights:
        print("[train_ontology_lora] Merging LoRA weights into base model …")
        model = model.merge_and_unload()
        model.save_pretrained(str(out_dir / "merged"))
        tokenizer.save_pretrained(str(out_dir / "merged"))
        print(f"[train_ontology_lora] Merged model saved → {out_dir / 'merged'}")
    else:
        model.save_pretrained(str(out_dir / "lora_adapter"))
        tokenizer.save_pretrained(str(out_dir / "lora_adapter"))
        print(f"[train_ontology_lora] LoRA adapter saved → {out_dir / 'lora_adapter'}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[train_ontology_lora] Done.")


if __name__ == "__main__":
    main()
