#!/usr/bin/env bash
# ============================================================
#  SLURM job: Mistral-7B-Instruct QLoRA on hierarchical ontology JSONL (Unsloth).
#  Submit: sbatch training_scripts/slurm_ontology_sft.sh
# ============================================================

#SBATCH --job-name=ontology-sft-mistral
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=.slurm/ontology_sft.out
#SBATCH --error=.slurm/ontology_sft.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=darragh11dec@gmail.com

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

python -c "import unsloth" 2>/dev/null || {
    echo "[slurm] installing unsloth …"
    pip install unsloth_zoo trl "datasets>=2.18" --quiet
    pip install "unsloth[cu124]" --break-system-packages --quiet
}

ONTO="${ROOT}/data/ontology_ner_full_hierarchical_alpaca.jsonl"
if [[ ! -f "$ONTO" ]]; then
    echo "Missing ${ONTO}" >&2
    echo "Create it from Neo4j, e.g.:" >&2
    echo "  PYTHONPATH=. python tools/data/export_full_ontology_ner_sft_jsonl.py --prompt-style alpaca --out data/ontology_ner_full_hierarchical_alpaca.jsonl" >&2
    exit 1
fi

echo "[slurm] training Mistral-7B-Instruct on ontology (Alpaca hierarchical text field) …"
PYTHONPATH=. python training_scripts/ner/biomistral_ner_finetune_unsloth.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.2 \
    --ontology-only \
    --full-ontology-finetune \
    --lora-dir models/mistral-ontology-lora \
    --merged-dir models/mistral-ontology-merged

echo "Job finished: $(date)"
