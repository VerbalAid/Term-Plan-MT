# Fine-tuning Mistral-7B-Instruct for graph-aware medical translation

## What you are building

A version of Mistral-7B-Instruct that, given a French sentence + MedDRA context, outputs an English translation AND a structured term map preserving hierarchy level. This slots directly into your S2-S5 pipeline as a drop-in replacement.

---

### Alternative: ontology-only SFT (no segment sentences)

To avoid using §4.8 (or any eval) segments as training text, build instruction examples **only from Neo4j** (`:Concept` with `fr_label`, English `name`, `level`, and immediate parent via `BROADER_THAN`):

```bash
PYTHONPATH=. python tools/data/build_ontology_sft.py --output-dir data/sft/ --limit 5000
```

Writes `data/sft/ontology_train.jsonl` and `data/sft/ontology_val.jsonl` (stratified up to `limit // 5` concepts per level L1–L5, then 90/10 shuffle). Use those paths in `train_mistral_sft.py` on the cluster instead of sentence-based `train.jsonl` / `val.jsonl`.

---

## Step 1 — Generate training data (on your laptop, no GPU needed)

You already have everything needed: Neo4j with MedDRA, your segment JSONLs, your existing pipeline.

Run:

```bash
PYTHONPATH=. .venv/bin/python tools/data/export_full_ontology_ner_sft_jsonl.py \
  --segments data/section48/segments_ner_biollm.jsonl \
  --output data/sft/mistral_translation_sft.jsonl \
  --format alpaca
```

If that script doesn't produce the right format, tell Cursor to modify it to produce exactly this structure for each training example:

```json
{
  "instruction": "Translate the French medical sentence to English. Use the MedDRA ontology context to preserve terminology at the correct hierarchy level. Output the translation followed by a term map.",
  "input": "French: Une pneumopathie immuno-médiée a été observée.\n\nExtracted terms:\n- pneumopathie immuno-médiée → MedDRA ID: 10038695 | Level: LLT (L5)\n\nOntology context:\n- LLT: immune-mediated pneumonitis\n- PT: pneumonitis\n- HLT: lower respiratory tract disorders\n- SOC: respiratory, thoracic and mediastinal disorders",
  "output": "Translation: Immune-mediated pneumonitis was observed.\n\nTerm map:\n- pneumopathie immuno-médiée → immune-mediated pneumonitis | MedDRA L5 | ID: 10038695"
}
```

You need at least 500 examples. Your 127 segments × multiple terms each should give you enough. Augment by using all three NER segment files (baseline, biollm, unsloth_full) to get more variety.

Split 90/10 train/val. Save as:

- `data/sft/train.jsonl`
- `data/sft/val.jsonl`

Copy both files to the cluster.

---

## Step 2 — Set up the cluster environment

SSH into the cluster and run:

```bash
# Request a GPU node — adjust to your cluster's scheduler
srun --gpus=1 --mem=40G --time=08:00:00 --pty bash
# or for SLURM:
# sbatch your_job.sh

# Create environment
python -m venv venv_termplan
source venv_termplan/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl datasets bitsandbytes accelerate
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

---

## Step 3 — Write the training script

Create `train_mistral_sft.py` on the cluster:

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.2"
TRAIN_FILE     = "data/sft/train.jsonl"
VAL_FILE       = "data/sft/val.jsonl"
OUTPUT_DIR     = "models/mistral-termplan-lora"
MAX_SEQ_LEN    = 1024
EPOCHS         = 3
BATCH_SIZE     = 2
GRAD_ACCUM     = 8   # effective batch = 16
LR             = 2e-4

# ── Load model in 4-bit ─────────────────────────────────────────────────────
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
)

# ── LoRA ────────────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── Data ────────────────────────────────────────────────────────────────────
def format_example(ex):
    return (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n{ex['output']}"
    )

dataset = load_dataset("json", data_files={
    "train": TRAIN_FILE,
    "validation": VAL_FILE,
})
dataset = dataset.map(lambda ex: {"text": format_example(ex)})

# Train on response only — critical for quality
collator = DataCollatorForCompletionOnlyLM(
    response_template="\n### Response:\n",
    tokenizer=tokenizer,
)

# ── Train ────────────────────────────────────────────────────────────────────
cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    bf16=True,
    fp16=False,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collator,
    args=cfg,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")
```

---

## Step 4 — Run training

```bash
python train_mistral_sft.py 2>&1 | tee training.log
```

Expected time on a cluster GPU (A100 40GB): 1-2 hours for 3 epochs on ~500 examples. On a V100: 3-4 hours.

Watch for:

- Loss starting around 2.0 and dropping to below 0.5 by epoch 3 — good
- Loss staying above 1.5 — not enough data or learning rate too high
- OOM — reduce `BATCH_SIZE` to 1 and increase `GRAD_ACCUM` to 16

---

## Step 5 — Copy model back and test

```bash
# On cluster
scp -r models/mistral-termplan-lora your_laptop:~/Desktop/Masters/MT/MT_Project_Terminology/models/

# On laptop — test CCR
export PYTHONPATH=.
.venv/bin/python tools/eval/evaluate.py --no-graph \
  --results-dir results/ner_biollm \
  --segments data/section48/segments_ner_biollm.jsonl
```

To use the fine-tuned model in your pipeline, modify `pipeline/systems/models.py` to add a `load_mistral_termplan()` function that loads `models/mistral-termplan-lora` with the same 4-bit config as `load_mistral_4bit()`. Then run S2-S5 with this model and compare HypRefAg.

---

## What success looks like

- HypRefAg increases on S2-S5 compared to base Mistral
- Term map output is consistently structured (you can parse it)
- Level distribution shows fewer concept flattening events

If HypRefAg doesn't improve meaningfully, that's also a valid result — it means the bottleneck is CCR (graph coverage) not the model's ability to use ontology context. Either way you have something to write about.
