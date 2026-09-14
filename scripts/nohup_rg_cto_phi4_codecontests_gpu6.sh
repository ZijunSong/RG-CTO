#!/bin/bash
# CodeContests × Phi-4-Reasoning × RG-CTO: 1 run on GPU6, reuse iter0 through iter2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES=6
export TENSOR_PARALLEL_SIZE=1
export DATASET=CodeContests
export QUESTION_FILE="${PROJECT_ROOT}/data/CodeContests_Test_165.jsonl"
export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Phi-4-reasoning}"

RUNS_ROOT="${PROJECT_ROOT}/results/runs/CodeContests_Phi_4_Reasoning_RG_CTO"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$RUNS_ROOT" "$LOG_DIR" /data/ppnm/tmp

END_INDEX=165
OUT_PREFIX="${RUNS_ROOT}/run0"
STEP1_RESULTS_SRC="${PROJECT_ROOT}/results/runs/CodeContests_Phi_4_Reasoning_RSE/run0_step1/results"

export OUT_PREFIX STEP1_RESULTS_SRC QUESTION_FILE DATASET MODEL_NAME
export RGCTO_GPU_MEMORY_UTILIZATION="${RGCTO_GPU_MEMORY_UTILIZATION:-0.70}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "========== CodeContests Phi-4 RG-CTO iter0→iter2 | GPU6 =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "STEP1_RESULTS_SRC=${STEP1_RESULTS_SRC}"
echo "QUESTION_FILE=${QUESTION_FILE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "RGCTO_GPU_MEMORY_UTILIZATION=${RGCTO_GPU_MEMORY_UTILIZATION} DISTILL_GPU_MEMORY_UTILIZATION=${DISTILL_GPU_MEMORY_UTILIZATION}"
echo "Started at $(date '+%F %T')"

bash "${SCRIPT_DIR}/run_rg_cto_phi4_from_iter0_iter2.sh" 0 "$END_INDEX"

echo ""
echo "========== CodeContests Phi-4 RG-CTO done at $(date '+%F %T') =========="
