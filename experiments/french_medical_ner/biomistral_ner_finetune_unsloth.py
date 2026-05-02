#!/usr/bin/env python3
"""LoRA SFT for BioMistral NER (Alpaca JSON targets) with Unsloth + TRL."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = ROOT / "scripts"
_NER_AB = ROOT / "experiments" / "french_medical_ner"
for _p in (ROOT, _SCRIPTS, _NER_AB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline.cuda_ld_path import ensure_cuda_pip_libs_visible

ensure_cuda_pip_libs_visible()

# Text-only LoRA: avoid importing torchvision. HF sets ``_torchvision_available`` from metadata even when
# ``import torchvision`` crashes with torch/torchvision ABI mismatch (e.g. RuntimeError: torchvision::nms).
try:
    import transformers.utils.import_utils as _hf_import_utils

    _hf_import_utils._torchvision_available = False
except Exception:
    pass


def _import_torch_cuda_first() -> None:
    """Fail fast when CUDA/NCCL libs mismatch (e.g. ``undefined symbol: ncclDevCommDestroy``)."""
    try:
        import torch  # noqa: F401
    except Exception as e:
        chain = f"{e!s} {getattr(e, '__cause__', '')!s}"
        low = chain.lower()
        if "nccl" in low or "undefined symbol" in low or "libtorch_cuda" in low:
            py = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise SystemExit(
                "ERROR: PyTorch failed to load CUDA (NCCL / symbol mismatch). Common fixes:\n\n"
                "1) Wrong NCCL on PYTHONPATH/LD path — pip's ``nvidia-nccl-cu12`` often conflicts with "
                "CUDA 13 builds (``torch …+cu130``). Try removing it so PyTorch uses its bundled NCCL:\n"
                "     pip uninstall -y nvidia-nccl-cu12\n"
                "   Then: python -c \"import torch; print(torch.__version__)\"\n\n"
                "2) Python " + py + ": PyTorch may have **no** wheels on ``cu124`` — use the CUDA index "
                "that matches the CUDA/driver stack (driver shows CUDA 13 → often ``cu130``):\n"
                "     pip install --force-reinstall torch torchvision torchaudio \\\n"
                "       --index-url https://download.pytorch.org/whl/cu130\n\n"
                "3) Prefer versions from https://pytorch.org/get-started/locally/ (same cu* as the GPU stack).\n\n"
                "4) ``unsloth-zoo`` may pin ``torch<2.11``; if pip warns, use torch 2.10.x with the same cu* URL "
                "or upgrade Unsloth when supported.\n\n"
                f"Underlying error: {e!r}"
            ) from e
        raise


_import_torch_cuda_first()


def _patch_datasets_fingerprint_if_py314() -> None:
    """HF ``datasets`` fingerprint pickle breaks on Python 3.14 + dill (same workaround as token HF trainers)."""
    if sys.version_info < (3, 14):
        return
    import hashlib

    import datasets.arrow_dataset as ads
    import datasets.fingerprint as fp
    import pyarrow as pa

    _orig = fp.generate_fingerprint

    def _safe_generate_fingerprint(dataset):
        try:
            return _orig(dataset)
        except TypeError as e:
            if "_batch_setitems" not in str(e):
                raise
            table = dataset.data.table
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()

    fp.generate_fingerprint = _safe_generate_fingerprint  # type: ignore[assignment]
    ads.generate_fingerprint = _safe_generate_fingerprint  # type: ignore[assignment]


def _dataset_from_text_rows(rows: list[dict[str, str]]):
    """Avoid ``Dataset.from_list`` on Py 3.14 (same fingerprint issue as CamemBERT fine-tune)."""
    from datasets import Dataset, Features, Value

    return Dataset.from_dict(
        {"text": [r["text"] for r in rows]},
        features=Features({"text": Value("string")}),
    )


def _python_dev_headers_present() -> bool:
    """Triton builds a tiny helper with gcc and needs ``Python.h`` (Fedora: ``python3-devel``)."""
    import sysconfig

    return (Path(sysconfig.get_path("include")) / "Python.h").is_file()


def _apply_unsloth_torch_rms_fallback(reason: str) -> None:
    """Replace Unsloth's Triton RMSNorm with pure PyTorch (slower, no gcc/Python.h)."""
    import torch

    def _fn(layernorm, X, gemma: bool = False):
        input_dtype = X.dtype
        x = X.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        eps = (
            layernorm.variance_epsilon
            if hasattr(layernorm, "variance_epsilon")
            else getattr(layernorm, "eps", 1e-6)
        )
        x = x * torch.rsqrt(variance + eps)
        if gemma:
            x = x * (layernorm.weight.to(torch.float32) + 1.0)
        else:
            x = x * layernorm.weight.to(torch.float32)
        return x.to(input_dtype)

    import unsloth.kernels.rms_layernorm as _rms_mod
    import unsloth.models.llama as _ullama

    _rms_mod.fast_rms_layernorm = _fn
    _ullama.fast_rms_layernorm = _fn
    try:
        import unsloth.models.mistral as _umistral

        _umistral.fast_rms_layernorm = _fn
    except Exception:
        pass
    try:
        import unsloth.kernels as _ukern

        if hasattr(_ukern, "fast_rms_layernorm"):
            _ukern.fast_rms_layernorm = _fn
    except Exception:
        pass
    print(f"Unsloth: PyTorch RMSNorm fallback enabled ({reason}).", file=sys.stderr)


def _dir_has_hf_model_weights(d: Path) -> bool:
    """True if ``d`` looks like a HF model checkpoint (not tokenizer-only)."""
    if not d.is_dir():
        return False
    if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
        return True
    return any(d.glob("model-*.safetensors"))


def _merge_lora_peft_fallback(
    base_model_id: str,
    lora_dir: Path,
    merged_dir: Path,
    tokenizer,
) -> None:
    """When Unsloth ``save_pretrained_merged`` writes only tokenizers, merge via PEFT (CPU by default)."""
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    preference = os.environ.get("UNSLOTH_MERGE_DEVICE", "cpu").strip().lower()
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if preference == "cuda" and torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = "cpu"

    print(
        f"Merging LoRA → dense weights via PEFT (device_map={device_map!r}; "
        f"UNSLOTH_MERGE_DEVICE=cuda to try GPU). Base={base_model_id!r}",
        file=sys.stderr,
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(lora_dir))
    merged_model = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))


try:
    from unsloth import FastLanguageModel
except ImportError as _e:
    msg = str(_e).lower()
    if "nccl" in msg or "undefined symbol" in msg or "libtorch_cuda" in msg:
        raise SystemExit(
            "ERROR: PyTorch CUDA failed while loading Unsloth (see NCCL note above if shown). "
            "Reinstall torch from https://pytorch.org/get-started/locally/ for the installed CUDA version."
        ) from _e
    raise ImportError(
        "Unsloth / dependencies missing. Install:\n"
        "  pip install unsloth_zoo trl 'datasets>=2.18'\n"
        "  pip install 'unsloth[cu124] @ git+https://github.com/unslothai/unsloth.git'\n"
        "(Use cu121/cu124/cu128 etc. to match the PyTorch CUDA build.)"
    ) from _e

_DEFAULT_BRAT = ROOT / "data" / "QUAERO_FrenchMed" / "corpus" / "train" / "EMEA"
_DEFAULT_LORA = ROOT / "models" / "biomistral-ner-lora"
_DEFAULT_MERGED = ROOT / "models" / "biomistral-ner-merged"
_BASE_MODEL = "BioMistral/BioMistral-7B"

ALPACA_INSTRUCTION = (
    "Extract all medical terms (disorders, drugs, procedures) from this French medical text. "
    "Return only a JSON list of strings."
)


def _load_plot_overlay():
    p = ROOT / "scripts" / "plot_training_curves.py"
    spec = importlib.util.spec_from_file_location("plot_training_curves", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.plot_training_curves_overlay


def merge_bio_to_entity_strings(tokens: list[str], tags: list[str]) -> list[str]:
    """Reconstruct entity strings from BIO tags (DISO/CHEM/PROC)."""
    out: list[str] = []
    cur: list[str] = []
    for tok, tag in zip(tokens, tags):
        if tag == "O":
            if cur:
                out.append(" ".join(cur))
                cur = []
            continue
        if tag.startswith("B-"):
            if cur:
                out.append(" ".join(cur))
            cur = [tok]
        elif tag.startswith("I-"):
            if cur:
                cur.append(tok)
            else:
                cur = [tok]
    if cur:
        out.append(" ".join(cur))
    return out


def to_alpaca_text(fr_sentence: str, json_list_str: str) -> str:
    return (
        "### Instruction:\n"
        f"{ALPACA_INSTRUCTION}\n\n"
        "### Input:\n"
        f"{fr_sentence}\n\n"
        "### Response:\n"
        f"{json_list_str}"
    )


def split_prompt_and_gold(full_text: str) -> tuple[str, list[str]]:
    marker = "### Response:\n"
    i = full_text.rfind(marker)
    if i == -1:
        raise ValueError("Alpaca example missing '### Response:' marker")
    prefix = full_text[: i + len(marker)]
    tail = full_text[i + len(marker) :].strip()
    gold = json.loads(tail)
    if not isinstance(gold, list):
        raise ValueError("Gold response is not a JSON list")
    return prefix, [str(x).strip() for x in gold if str(x).strip()]


def main() -> None:
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit("Install trl and datasets: pip install trl datasets") from e

    from collections import Counter

    from quaero_brat_reader import ID2LABEL, load_quaero_brat
    from biomistral_prompt_ner import parse_llm_terms

    def multiset_f1(gold: list[str], pred_terms: list[dict]) -> float:
        pred_strings = [
            str(t["word"]).strip() for t in pred_terms if t and str(t.get("word", "")).strip()
        ]
        cg = Counter(gold)
        cp = Counter(pred_strings)
        inter = sum((cg & cp).values())
        if inter == 0:
            return 0.0
        p = inter / max(sum(cp.values()), 1)
        r = inter / max(sum(cg.values()), 1)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    p = argparse.ArgumentParser(description="Unsloth LoRA SFT for BioMistral JSON-list NER.")
    p.add_argument("--brat-dir", type=Path, default=_DEFAULT_BRAT, help="QUAERO BRAT folder (.txt + .ann).")
    p.add_argument("--lora-dir", type=Path, default=_DEFAULT_LORA, help="LoRA adapter + checkpoints.")
    p.add_argument("--merged-dir", type=Path, default=_DEFAULT_MERGED, help="Merged full-precision export.")
    p.add_argument("--base-model", type=str, default=_BASE_MODEL, help="HF base model id.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-combine-medline",
        action="store_true",
        help="Do not also load sibling MEDLINE/ (same default as ``quaero_brat_reader.load_quaero_brat``).",
    )
    p.add_argument(
        "--limit-records",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N BRAT sentences after shuffle (debug / smoke test).",
    )
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load the base model in bf16/fp16 instead of 4-bit (more VRAM). Use when bitsandbytes fails "
        "(e.g. missing libnvJitLink.so.* on the loader path).",
    )
    p.add_argument(
        "--low-vram",
        action="store_true",
        help="Cap context at 192, smaller LoRA (r=8), and minimal eval generation — for ~8 GB laptop GPUs.",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        metavar="L",
        help="Training / eval context length (lower = less VRAM; default 256 for ~8 GB).",
    )
    p.add_argument(
        "--lora-r",
        type=int,
        default=16,
        metavar="R",
        help="LoRA rank (default 16; use 8 with --low-vram or on GPU OOM).",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        metavar="A",
        help="LoRA alpha (default 32; often 2× rank).",
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (0 saves a bit of memory with Unsloth fast path).",
    )
    p.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Micro-batch per GPU (default 1 for tight VRAM).",
    )
    p.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Eval micro-batch per GPU (default 1).",
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        metavar="N",
        help="Gradient accumulation steps (default 16 → effective batch 16 with batch size 1).",
    )
    p.add_argument(
        "--force-torch-rms",
        action="store_true",
        help="Use PyTorch RMSNorm instead of Unsloth Triton kernels (slower; avoids needing Python.h).",
    )
    p.add_argument(
        "--allow-triton-rms",
        action="store_true",
        help="Keep Unsloth Triton RMSNorm (requires Python dev headers for first-time Triton driver build).",
    )
    p.add_argument(
        "--export-merged-only",
        action="store_true",
        help="Skip training: merge existing adapter in --lora-dir into --merged-dir using PEFT (needs RAM for 7B).",
    )
    args = p.parse_args()

    if args.force_torch_rms:
        _apply_unsloth_torch_rms_fallback("CLI --force-torch-rms")
    elif not args.allow_triton_rms and not _python_dev_headers_present():
        _apply_unsloth_torch_rms_fallback(
            "Python.h not found — install ``python3-devel`` (Fedora) for fast Triton RMS, or pass ``--force-torch-rms``"
        )

    lora_r = max(1, int(args.lora_r))
    lora_alpha = max(1, int(args.lora_alpha))
    lora_dropout = float(args.lora_dropout)
    max_seq_user = max(64, int(args.max_seq_length))
    if args.low_vram:
        max_seq_length_run = min(max_seq_user, 192)
        lora_r = min(lora_r, 8)
        lora_alpha = min(lora_alpha, 16)
        lora_dropout = 0.0
    else:
        max_seq_length_run = max_seq_user

    _patch_datasets_fingerprint_if_py314()

    brat_dir = args.brat_dir if args.brat_dir.is_absolute() else ROOT / args.brat_dir
    lora_dir = args.lora_dir if args.lora_dir.is_absolute() else ROOT / args.lora_dir
    merged_dir = args.merged_dir if args.merged_dir.is_absolute() else ROOT / args.merged_dir
    lora_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    if args.export_merged_only:
        from transformers import AutoTokenizer

        if not lora_dir.is_dir():
            raise SystemExit(f"--export-merged-only: LoRA dir not found: {lora_dir}")
        tok = AutoTokenizer.from_pretrained(str(lora_dir))
        try:
            _merge_lora_peft_fallback(args.base_model, lora_dir, merged_dir, tok)
        except Exception as ex:
            raise SystemExit(
                "PEFT merge failed (needs enough CPU RAM for 7B, or try UNSLOTH_MERGE_DEVICE=cuda).\n"
                f"{ex}"
            ) from ex
        if not _dir_has_hf_model_weights(merged_dir):
            raise SystemExit(f"Merge produced no weight files under {merged_dir}")
        print(f"Merged HF weights: {merged_dir}", file=sys.stderr)
        return

    records = load_quaero_brat(brat_dir, combine_medline=not args.no_combine_medline)
    if args.limit_records is not None:
        lim = max(1, int(args.limit_records))
        records = records[:lim]
    rng = random.Random(args.seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n = len(records)
    if n <= 1:
        train_recs = list(records)
        val_recs = list(records)
    else:
        n_train = max(1, int(0.9 * n))
        if n_train >= n:
            n_train = n - 1
        train_recs = [records[i] for i in idx[:n_train]]
        val_recs = [records[i] for i in idx[n_train:]]

    def record_to_text(rec: dict) -> str:
        tokens = rec["tokens"]
        tags = [ID2LABEL[int(i)] for i in rec["ner_tags"]]
        fr = " ".join(tokens)
        entities = merge_bio_to_entity_strings(tokens, tags)
        payload = json.dumps(entities, ensure_ascii=False)
        return to_alpaca_text(fr, payload)

    train_rows = [{"text": record_to_text(r)} for r in train_recs]
    val_rows = [{"text": record_to_text(r)} for r in val_recs]
    train_ds = _dataset_from_text_rows(train_rows)
    val_ds = _dataset_from_text_rows(val_rows)

    import torch

    max_seq_length = max_seq_length_run
    load_in_4bit = not args.no_4bit
    if load_in_4bit:
        dtype = None
    else:
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Unsloth keeps compute weights in bf16 when supported; mixed precision must match (fp16 + bf16 model → TypeError).
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    cfg_kwargs: dict = {
        "output_dir": str(lora_dir),
        "num_train_epochs": 3,
        "per_device_train_batch_size": max(1, int(args.per_device_train_batch_size)),
        "per_device_eval_batch_size": max(1, int(args.per_device_eval_batch_size)),
        "gradient_accumulation_steps": max(1, int(args.gradient_accumulation_steps)),
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "logging_steps": 10,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "seed": args.seed,
    }

    sig = inspect.signature(SFTConfig.__init__)
    if "completion_only_loss" in sig.parameters:
        cfg_kwargs["completion_only_loss"] = True
    elif "train_on_responses_only" in sig.parameters:
        cfg_kwargs["train_on_responses_only"] = True
    if "response_template" in sig.parameters:
        cfg_kwargs["response_template"] = "\n### Response:\n"
    if "max_seq_length" in sig.parameters:
        cfg_kwargs["max_seq_length"] = max_seq_length
    if "packing" in sig.parameters:
        cfg_kwargs["packing"] = False

    if "eval_strategy" not in sig.parameters and "evaluation_strategy" in sig.parameters:
        cfg_kwargs["evaluation_strategy"] = cfg_kwargs.pop("eval_strategy")

    filtered = {k: v for k, v in cfg_kwargs.items() if k in sig.parameters}
    training_args = SFTConfig(**filtered)

    eval_ner_cap = 4 if args.low_vram else 16
    eval_max_new = min(96, max(32, max_seq_length // 2))

    class NerSFTTrainer(SFTTrainer):
        """Adds multiset JSON-list F1 on the validation set each evaluation."""

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            import torch

            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if metrics is None:
                metrics = {}
            ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            if ds is None:
                return metrics

            eval_ds_list = val_rows  # closure: same order as val_ds

            self.model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            if proc is None:
                return metrics
            scores: list[float] = []
            cap = min(eval_ner_cap, len(eval_ds_list))
            for i in range(cap):
                full = eval_ds_list[i]["text"]
                prefix, gold_list = split_prompt_and_gold(full)
                inputs = proc(
                    prefix,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_length,
                )
                dev = next(self.model.parameters()).device
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                with torch.inference_mode():
                    out_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=eval_max_new,
                        do_sample=False,
                        pad_token_id=getattr(proc, "pad_token_id", None) or proc.eos_token_id,
                        use_cache=False,
                    )
                in_len = inputs["input_ids"].shape[1]
                gen_text = proc.decode(
                    out_ids[0, in_len:],
                    skip_special_tokens=True,
                )
                pt = parse_llm_terms(gen_text)
                pred_terms = pt if pt else []
                scores.append(multiset_f1(gold_list, pred_terms))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if scores:
                metrics["eval_ner_f1"] = sum(scores) / len(scores)
            self.model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return metrics

    tr_sig = inspect.signature(SFTTrainer.__init__)
    trainer_kw: dict = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "args": training_args,
        "dataset_text_field": "text",
    }
    if "processing_class" in tr_sig.parameters:
        trainer_kw["processing_class"] = tokenizer
    else:
        trainer_kw["tokenizer"] = tokenizer
    if "max_seq_length" not in sig.parameters and "max_seq_length" in tr_sig.parameters:
        trainer_kw["max_seq_length"] = max_seq_length

    trainer = NerSFTTrainer(**trainer_kw)

    trainer.train()

    # Save adapter + tokenizer (root of lora_dir)
    trainer.save_model(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    # Merge LoRA into dense weights for downstream HF / 4-bit load in biomistral_prompt_ner.py
    merged_dir.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(model, "save_pretrained_merged"):
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    except Exception as ex:
        print(f"warning: save_pretrained_merged failed ({ex}); will try PEFT fallback if needed.", file=sys.stderr)
    tokenizer.save_pretrained(str(merged_dir))

    if not _dir_has_hf_model_weights(merged_dir):
        print(
            "Merged export has no weight files (tokenizer-only). Running PEFT merge_and_unload fallback…",
            file=sys.stderr,
        )
        try:
            _merge_lora_peft_fallback(args.base_model, lora_dir, merged_dir, tokenizer)
        except Exception as ex:
            raise SystemExit(
                "Could not create merged weight files under:\n"
                f"  {merged_dir}\n"
                "Your LoRA adapter is saved here (use with biomistral_prompt_ner --backend unsloth --unsloth-lora-path):\n"
                f"  {lora_dir}\n"
                "PEFT merge needs enough RAM (often ~16 GB+ system RAM for 7B on CPU). "
                "Optional: UNSLOTH_MERGE_DEVICE=cuda if GPU fits bf16 merge.\n"
                f"Error: {ex}"
            ) from ex

    if not _dir_has_hf_model_weights(merged_dir):
        raise SystemExit(
            f"Merged dir still has no model weights after fallback: {merged_dir}\n"
            f"Use LoRA adapter at {lora_dir} with: biomistral_prompt_ner --backend unsloth --unsloth-lora-path ..."
        )

    # Training curves (this run only; use plot_training_curves.py CLI to overlay CamemBERT, etc.)
    try:
        plot_overlay = _load_plot_overlay()
        plot_overlay([lora_dir], lora_dir / "training_curves.png")
    except Exception as ex:
        print(f"warning: could not plot training curves: {ex}", file=sys.stderr)

    print(f"LoRA adapter: {lora_dir}")
    print(f"Merged model: {merged_dir}")


if __name__ == "__main__":
    main()
