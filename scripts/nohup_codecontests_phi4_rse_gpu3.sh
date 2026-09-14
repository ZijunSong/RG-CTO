#!/bin/bash
# CodeContests × Phi-4-Reasoning × RSE: 1 full run on GPU3, iter0 through iter2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES=3
export TENSOR_PARALLEL_SIZE=1
export DATASET=CodeContests
export QUESTION_FILE="${PROJECT_ROOT}/data/CodeContests_Test_165.jsonl"
export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Phi-4-reasoning}"

RUNS_ROOT="${PROJECT_ROOT}/results/runs/CodeContests_Phi_4_Reasoning_RSE"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$RUNS_ROOT" "$LOG_DIR" /data/ppnm/tmp

END_INDEX=165
OUT_PREFIX="${RUNS_ROOT}/run0"

echo "========== CodeContests Phi-4 RSE iter0→iter2 | GPU3 =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "QUESTION_FILE=${QUESTION_FILE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "Started at $(date '+%F %T')"

export OUT_PREFIX QUESTION_FILE DATASET MODEL_NAME
bash "${SCRIPT_DIR}/run_rse_phi4_codecontests_to_iter2.sh" 0 "$END_INDEX"

echo ""
echo "========== CodeContests Phi-4 RSE done at $(date '+%F %T') =========="
