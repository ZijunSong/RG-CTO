#!/bin/bash
# BambooQA × Phi-4-Reasoning × RG-CTO: 3 runs on GPU2, shared iter0, iter1-2 only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "${SCRIPT_DIR}/common/nohup_helpers.sh"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES=2
export TENSOR_PARALLEL_SIZE=1
export DATASET=BambooQA
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RGCTO_GPU_MEMORY_UTILIZATION="${RGCTO_GPU_MEMORY_UTILIZATION:-0.75}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.78}"
mkdir -p /data/ppnm/tmp
export TMPDIR="/data/ppnm/tmp/rg_cto_phi4_gpu2_$$"

RUNS_ROOT="${PROJECT_ROOT}/results/runs/BambooQA_Phi_4_Reasoning_RG_CTO_gpu2"
STEP1_RESULTS_SRC="${STEP1_RESULTS_SRC:-${PROJECT_ROOT}/results/iter0/BambooQA/Phi_4_Reasoning/results}"
export QUESTION_FILE="${PROJECT_ROOT}/data/BambooQA.jsonl"
END_INDEX=125
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$RUNS_ROOT" "$LOG_DIR"

export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.78}"

echo "========== BambooQA Phi-4 RG-CTO 3× | GPU2 | shared iter0 =========="
echo "RUNS_ROOT=${RUNS_ROOT}"
echo "STEP1_RESULTS_SRC=${STEP1_RESULTS_SRC}"
echo "Started at $(date '+%F %T')"

for run_id in 0 1 2; do
  OUT_PREFIX="${RUNS_ROOT}/run${run_id}"
  if run_has_valid_summary "${OUT_PREFIX}"; then
    echo "[skip] run${run_id} already has valid eval summary"
    continue
  fi
  echo ""
  echo ">>>>>>>>>> Run ${run_id}/2 <<<<<<<<<<"
  export OUT_PREFIX STEP1_RESULTS_SRC QUESTION_FILE DATASET
  bash "${SCRIPT_DIR}/run_rg_cto_phi4_from_iter0_iter2.sh" 0 "$END_INDEX"
done

python "${SCRIPT_DIR}/patch_rg_cto_phi4_table.py" || true

echo ""
echo "========== BambooQA Phi-4 RG-CTO (GPU2) done at $(date '+%F %T') =========="
