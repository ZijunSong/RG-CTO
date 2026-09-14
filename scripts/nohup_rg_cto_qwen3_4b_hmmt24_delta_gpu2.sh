#!/bin/bash
# HMMT24 × Qwen3-4B-Thinking × RG-CTO (delta=0.6): 1 run on GPU2, iter0→iter2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES=2
export TENSOR_PARALLEL_SIZE=1
export DATASET=HMMT24
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Qwen3-4B-Thinking-2507}"
export QUESTION_FILE="${PROJECT_ROOT}/data/HMMT_24.jsonl"
export STEP1_RESULTS_SRC="${STEP1_RESULTS_SRC:-${PROJECT_ROOT}/results/iter0/HMMT24/Qwen3_4B_Thinking_2507/results}"

# run_experiment.sh 对齐超参
export ALPHA=0.55
export GATE_DELTA=0.6
export THRESHOLD=0.8
export BATCH_SIZE=16
export N_EXP_COMPLETIONS=48
export CTO_RETRIEVAL_RERANK_POOL_MULT=8
export RGCTO_GPU_MEMORY_UTILIZATION="${RGCTO_GPU_MEMORY_UTILIZATION:-0.60}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"

RUN_LABEL="${RUN_LABEL:-seed1}"
OUT_PREFIX="${PROJECT_ROOT}/results/runs/hmmt24_rg_cto_qwen3_4b_delta_0_6/${RUN_LABEL}"
END_INDEX=30

mkdir -p "${PROJECT_ROOT}/logs" /data/ppnm/tmp
export OUT_PREFIX STEP1_RESULTS_SRC QUESTION_FILE DATASET MODEL_NAME

echo "========== HMMT24 Qwen3-4B RG-CTO delta=0.6 | GPU2 | ${RUN_LABEL} =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "STEP1_RESULTS_SRC=${STEP1_RESULTS_SRC}"
echo "ALPHA=${ALPHA} GATE_DELTA=${GATE_DELTA} THRESHOLD=${THRESHOLD} BATCH_SIZE=${BATCH_SIZE}"
echo "Started at $(date '+%F %T')"

bash "${SCRIPT_DIR}/run_rg_cto_qwen3_4b_from_iter0_iter2.sh" 0 "$END_INDEX"

echo ""
echo "========== Done delta=0.6 GPU2 at $(date '+%F %T') =========="
