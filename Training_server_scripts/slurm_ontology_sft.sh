#!/usr/bin/env bash
# ============================================================
#  SLURM job: MedDRA ontology LoRA fine-tune on AZTI
#  Server: azti.hitz.eus  (10x L40S, 45 GB VRAM each)
#
#  Submit: sbatch experiments/slurm_ontology_sft.sh
#  Watch:  watch squeue
#  Logs:   .slurm/ontology_sft.out / .slurm/ontology_sft.err
# ============================================================

#SBATCH --job-name=termplan-ontology-sft
#SBATCH --cpus-per-task=4          # 4 CPU cores for data loading
#SBATCH --gres=gpu:1               # 1x L40S is enough for 4-bit Mistral 7B
#SBATCH --mem=32G                  # 32 GB RAM (model + data buffers)
#SBATCH --time=08:00:00            # 8 h wall-clock limit (adjust if needed)
#SBATCH --output=.slurm/ontology_sft.out
#SBATCH --error=.slurm/ontology_sft.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=darragh11dec@gmail.com   # required by GPU_nice.pdf rules

# ---- Ensure log directory exists ----
mkdir -p .slurm

echo "========================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================================"

# ---- Environment ----
# Option A: use the shared HiTZ transformers environment (fastest to start)
source /var/python3envs/transformers-5.3.0/bin/activate

# Option B (if A doesn't have unsloth): activate your own venv instead
# source ~/envs/termplan/bin/activate

# Shared model cache — avoids re-downloading Mistral every run
export TRANSFORMERS_CACHE="/tcache"
export HF_HOME="/tcache"

# Reduce tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# ---- Install unsloth if not present in the activated env ----
python -c "import unsloth" 2>/dev/null || {
    echo "[slurm] unsloth not found — installing into current env …"
    pip install unsloth_zoo --quiet
    pip install "unsloth[cu124]" --break-system-packages --quiet
}

# ---- Build SFT data if not already done ----
if [ ! -f "data/sft/ontology_train.jsonl" ]; then
    echo "[slurm] Building ontology SFT data …"
    PYTHONPATH=. python tools/data/build_ontology_sft.py \
        --out-train data/sft/ontology_train.jsonl \
        --out-val   data/sft/ontology_val.jsonl
fi

# ---- Train ----
echo "[slurm] Starting LoRA fine-tune …"
PYTHONPATH=. python experiments/train_ontology_lora.py \
    --train  data/sft/ontology_train.jsonl \
    --val    data/sft/ontology_val.jsonl \
    --out    models/mistral-meddra-lora \
    --epochs 3 \
    --batch  4 \
    --grad-acc 4 \
    --lora-r 16 \
    --lora-alpha 32

# Use --merge-weights if you want a standalone model (no PEFT dependency at inference)
# --merge-weights produces models/mistral-meddra-lora/merged/

echo "========================================================"
echo "Job finished: $(date)"
echo "========================================================"
