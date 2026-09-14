#!/bin/bash
# HMMT24 × Qwen3-4B-Thinking × RG-CTO (lambda_l=0.25): 1 run on GPU1, iter0→iter2.
# GPU3 被 CodeContests RSE 占用，放到空闲 GPU1。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES=1
export TENSOR_PARALLEL_SIZE=1
export DATASET=HMMT24
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Qwen3-4B-Thinking-2507}"
export QUESTION_FILE="${PROJECT_ROOT}/data/HMMT_24.jsonl"
export STEP1_RESULTS_SRC="${STEP1_RESULTS_SRC:-${PROJECT_ROOT}/results/iter0/HMMT24/Qwen3_4B_Thinking_2507/results}"

# 默认超参，只改 lambda_l
export ALPHA=0.55
export GATE_DELTA=0.4
export THRESHOLD=0.8
export LAMBDA_U=0.5
export LAMBDA_L=0.25
export K_CTO=8
export BATCH_SIZE=16
export N_EXP_COMPLETIONS=48
export CTO_RETRIEVAL_RERANK_POOL_MULT=8
export RGCTO_GPU_MEMORY_UTILIZATION="${RGCTO_GPU_MEMORY_UTILIZATION:-0.60}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"

RUN_LABEL="${RUN_LABEL:-seed1}"
OUT_PREFIX="${PROJECT_ROOT}/results/runs/hmmt24_rg_cto_qwen3_4b_lambda_l_0_25/${RUN_LABEL}"
END_INDEX=30

mkdir -p "${PROJECT_ROOT}/logs" /data/ppnm/tmp
export OUT_PREFIX STEP1_RESULTS_SRC QUESTION_FILE DATASET MODEL_NAME

echo "========== HMMT24 Qwen3-4B RG-CTO lambda_l=0.25 | GPU1 | ${RUN_LABEL} =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "ALPHA=${ALPHA} GATE_DELTA=${GATE_DELTA} THRESHOLD=${THRESHOLD} LAMBDA_U=${LAMBDA_U} LAMBDA_L=${LAMBDA_L} K_CTO=${K_CTO} BATCH_SIZE=${BATCH_SIZE}"
echo "Started at $(date '+%F %T')"

bash "${SCRIPT_DIR}/run_rg_cto_qwen3_4b_from_iter0_iter2.sh" 0 "$END_INDEX"

echo ""
echo "========== Done lambda_l=0.25 GPU1 at $(date '+%F %T') =========="
