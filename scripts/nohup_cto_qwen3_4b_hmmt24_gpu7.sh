#!/bin/bash
# HMMT24 × Qwen3-4B-Thinking × CTO: 1 run on GPU7, shared iter0, iter1-2 only.
# Memory set to 0.5 because GPU7 is shared with other users' processes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES=7
export TENSOR_PARALLEL_SIZE=1
export DATASET=HMMT24
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CTO_GPU_MEMORY_UTILIZATION="${CTO_GPU_MEMORY_UTILIZATION:-0.50}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.50}"

export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Qwen3-4B-Thinking-2507}"
export QUESTION_FILE="${PROJECT_ROOT}/data/HMMT_24.jsonl"

export ALPHA="${ALPHA:-0.55}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MAX_TOKENS="${MAX_TOKENS:-38912}"
export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-8192}"
export DISTILL_MAX_MODEL_LEN="${DISTILL_MAX_MODEL_LEN:-100000}"
export CTO_MAX_MODEL_LEN="${CTO_MAX_MODEL_LEN:-100000}"
export DISTILL_MAX_NUM_SEQS="${DISTILL_MAX_NUM_SEQS:-128}"

RUNS_ROOT="${PROJECT_ROOT}/results/runs/HMMT24_Qwen3_4B_Thinking_2507_CTO"
STEP1_RESULTS_SRC="${STEP1_RESULTS_SRC:-${PROJECT_ROOT}/results/iter0/HMMT24/Qwen3_4B_Thinking_2507/results}"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$RUNS_ROOT" "$LOG_DIR" /data/ppnm/tmp

END_INDEX=30
OUT_PREFIX="${RUNS_ROOT}/run0"

export OUT_PREFIX STEP1_RESULTS_SRC QUESTION_FILE DATASET MODEL_NAME

echo "========== HMMT24 Qwen3-4B CTO iter0→iter2 | GPU7 =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "STEP1_RESULTS_SRC=${STEP1_RESULTS_SRC}"
echo "QUESTION_FILE=${QUESTION_FILE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "ALPHA=${ALPHA} BATCH_SIZE=${BATCH_SIZE} MAX_TOKENS=${MAX_TOKENS} DISTILL_MAX_TOKENS=${DISTILL_MAX_TOKENS}"
echo "CTO_GPU_MEM=${CTO_GPU_MEMORY_UTILIZATION} DISTILL_GPU_MEM=${DISTILL_GPU_MEMORY_UTILIZATION}"
echo "Started at $(date '+%F %T')"

bash "${SCRIPT_DIR}/run_cto_qwen3_4b_from_iter0_iter2.sh" 0 "$END_INDEX"

echo ""
echo "========== HMMT24 Qwen3-4B CTO done at $(date '+%F %T') =========="
