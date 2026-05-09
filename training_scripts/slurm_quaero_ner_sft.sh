#!/usr/bin/env bash
# ============================================================
#  SLURM job: BioMistral QLoRA on QUAERO BRAT NER (Unsloth).
#  Submit: sbatch training_scripts/slurm_quaero_ner_sft.sh
# ============================================================
#
# Expected corpus path (default in trainer):
#   data/QUAERO_FrenchMed/corpus/train/EMEA
#
# This trains a NER model from QUAERO sentences only (no ontology SFT).

#SBATCH --job-name=quaero-ner-biomistral
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=.slurm/quaero_ner_sft.out
#SBATCH --error=.slurm/quaero_ner_sft.err

set -euo pipefail
mkdir -p .slurm

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Job started: $(date)  host=$(hostname)"

# Option A: shared env (adjust to your cluster)
# shellcheck disable=SC1091
source /var/python3envs/transformers-5.3.0/bin/activate

export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tcache}"
export HF_HOME="${HF_HOME:-/tcache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -c "import unsloth" 2>/dev/null || {
    echo "[slurm] installing unsloth …"
    pip install unsloth_zoo trl "datasets>=2.18" --quiet
    pip install "unsloth[cu124]" --break-system-packages --quiet
}

echo "[slurm] training BioMistral on QUAERO BRAT (DISO/CHEM/PROC) …"
PYTHONPATH=. python training_scripts/ner/biomistral_ner_finetune_unsloth.py \
    --base-model BioMistral/BioMistral-7B \
    --lora-dir models/biomistral-quaero-ner-lora \
    --merged-dir models/biomistral-quaero-ner-merged

echo "Job finished: $(date)"

