#!/bin/bash
# HMMT24 × Qwen3-4B-Thinking × RG-CTO (alpha_0=0.3): 1 run on GPU7, iter0→iter2.
# Aligns with run_experiment.sh: alpha_0=0.3, K_cto=8, tau_match=0.8,
# lambda_u=0.75, lambda_l=0.5, delta=0.4, MAX_TOKENS/max-pilot-tokens=38912.
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

export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Qwen3-4B-Thinking-2507}"
export QUESTION_FILE="${PROJECT_ROOT}/data/HMMT_24.jsonl"
export STEP1_RESULTS_SRC="${STEP1_RESULTS_SRC:-${PROJECT_ROOT}/results/iter0/HMMT24/Qwen3_4B_Thinking_2507/results}"

export ALPHA=0.3
export GATE_DELTA=0.4
export THRESHOLD=0.8
export LAMBDA_U=0.75
export LAMBDA_L=0.5
export K_CTO=8
export CANDIDATE_K=8
export BATCH_SIZE=8
export N_EXP_COMPLETIONS=48
export CTO_RETRIEVAL_RERANK_POOL_MULT=8

export MAX_TOKENS=38912
export MAX_PILOT_TOKENS=38912
export DISTILL_MAX_TOKENS=38912
export DISTILL_MAX_MODEL_LEN=100000
export RGCTO_MAX_MODEL_LEN=100000
export DISTILL_MAX_NUM_SEQS=128
export DISTILL_OVERSIZED_PROMPT_POLICY=truncate
export DISTILL_JSON_MAX_GEN=8192
export VLLM_SCORE_BATCH_SIZE=1
export RGCTO_GPU_MEMORY_UTILIZATION=0.5
export DISTILL_GPU_MEMORY_UTILIZATION=0.5

RUN_LABEL="${RUN_LABEL:-seed1}"
OUT_PREFIX="${PROJECT_ROOT}/results/runs/hmmt24_cto_4b_alpha0_03/${RUN_LABEL}"
END_INDEX=30

mkdir -p "${PROJECT_ROOT}/logs" /data/ppnm/tmp "${OUT_PREFIX%/*}"
export OUT_PREFIX STEP1_RESULTS_SRC QUESTION_FILE DATASET MODEL_NAME

echo "========== HMMT24 Qwen3-4B RG-CTO alpha_0=0.3 | GPU7 | ${RUN_LABEL} =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "ALPHA=${ALPHA} GATE_DELTA=${GATE_DELTA} THRESHOLD=${THRESHOLD} LAMBDA_U=${LAMBDA_U} LAMBDA_L=${LAMBDA_L} K_CTO=${K_CTO} BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_TOKENS=${MAX_TOKENS} MAX_PILOT_TOKENS=${MAX_PILOT_TOKENS} DISTILL_MAX_TOKENS=${DISTILL_MAX_TOKENS} DISTILL_JSON_MAX_GEN=${DISTILL_JSON_MAX_GEN}"
echo "GPU_MEM=${RGCTO_GPU_MEMORY_UTILIZATION} VLLM_SCORE_BATCH_SIZE=${VLLM_SCORE_BATCH_SIZE}"
echo "Started at $(date '+%F %T')"

bash "${SCRIPT_DIR}/run_rg_cto_qwen3_4b_from_iter0_iter2.sh" 0 "$END_INDEX"

echo ""
echo "========== Done alpha_0=0.3 GPU7 at $(date '+%F %T') =========="
