#!/bin/bash
# HLE_math_text × Qwen3-30B × RG-CTO: 3 runs on GPU7, shared best iter0, iter1-2 only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES=7
export TENSOR_PARALLEL_SIZE=1

RUNS_ROOT="${PROJECT_ROOT}/results/runs/HLE_math_text_Qwen3_30B_A3B_Thinking_2507_RG_CTO"
STEP1_RESULTS_SRC="${STEP1_RESULTS_SRC:-${PROJECT_ROOT}/results/iter0/HLE_math_text/Qwen3_30B_A3B_Thinking_2507/results}"
export QUESTION_FILE="${PROJECT_ROOT}/data/HLE_MATH_text_100_sample_subset.jsonl"
END_INDEX=100
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$RUNS_ROOT" "$LOG_DIR"

echo "========== HLE 30B RG-CTO 3× | GPU7 | shared iter0 =========="
echo "RUNS_ROOT=${RUNS_ROOT}"
echo "STEP1_RESULTS_SRC=${STEP1_RESULTS_SRC}"
echo "Started at $(date '+%F %T')"

for run_id in 0 1 2; do
  OUT_PREFIX="${RUNS_ROOT}/run${run_id}"
  if [ -f "${OUT_PREFIX}_eval_summary.json" ]; then
    echo "[skip] run${run_id} already has eval summary"
    continue
  fi
  echo ""
  echo ">>>>>>>>>> Run ${run_id}/2 <<<<<<<<<<"
  export OUT_PREFIX STEP1_RESULTS_SRC QUESTION_FILE
  bash "${SCRIPT_DIR}/run_rg_cto_from_iter0_iter2.sh" 0 "$END_INDEX"
done

python "${SCRIPT_DIR}/patch_rg_cto_30b_table.py" || true

echo ""
echo "========== HLE RG-CTO done at $(date '+%F %T') =========="
