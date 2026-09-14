#!/bin/bash
# HMMT25 × Qwen3-4B-Thinking × RG-CTO: 1 run on GPU7, shared iter0, iter1-2 only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES=7
export TENSOR_PARALLEL_SIZE=1
export DATASET=HMMT25
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RGCTO_GPU_MEMORY_UTILIZATION="${RGCTO_GPU_MEMORY_UTILIZATION:-0.60}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"

export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Qwen3-4B-Thinking-2507}"
export QUESTION_FILE="${PROJECT_ROOT}/data/HMMT_25.jsonl"

RUNS_ROOT="${PROJECT_ROOT}/results/runs/HMMT25_Qwen3_4B_Thinking_2507_RG_CTO"
STEP1_RESULTS_SRC="${STEP1_RESULTS_SRC:-/data/ppnm/Contrastive-Trajectory-Optimization/runs_method/HMMT25_Qwen3_4B_Thinking_2507_CTO_vllm_step1/results}"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$RUNS_ROOT" "$LOG_DIR" /data/ppnm/tmp

END_INDEX=30
OUT_PREFIX="${RUNS_ROOT}/run0"

export OUT_PREFIX STEP1_RESULTS_SRC QUESTION_FILE DATASET MODEL_NAME

echo "========== HMMT25 Qwen3-4B RG-CTO iter0→iter2 | GPU7 =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "STEP1_RESULTS_SRC=${STEP1_RESULTS_SRC}"
echo "QUESTION_FILE=${QUESTION_FILE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "Started at $(date '+%F %T')"

bash "${SCRIPT_DIR}/run_rg_cto_qwen3_4b_from_iter0_iter2.sh" 0 "$END_INDEX"

echo ""
echo "========== HMMT25 Qwen3-4B RG-CTO done at $(date '+%F %T') =========="
