#!/bin/bash
# Resume BambooQA × Phi-4 RG-CTO run0 on GPU{0,1,2} from step5 (iter2).
# Usage: bash scripts/resume_bambooqa_phi4_rg_cto_step5.sh <gpu_id>
set -euo pipefail

GPU_ID="${1:?Usage: $0 <gpu_id: 0|1|2>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TENSOR_PARALLEL_SIZE=1
export DATASET=BambooQA
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RGCTO_GPU_MEMORY_UTILIZATION="${RGCTO_GPU_MEMORY_UTILIZATION:-0.75}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.78}"
export TMPDIR="/data/ppnm/tmp/rg_cto_phi4_resume_gpu${GPU_ID}_$$"
mkdir -p /data/ppnm/tmp "$TMPDIR"

export QUESTION_FILE="${PROJECT_ROOT}/data/BambooQA.jsonl"
export STEP1_RESULTS_SRC="${PROJECT_ROOT}/results/iter0/BambooQA/Phi_4_Reasoning/results"
END_INDEX=125

case "$GPU_ID" in
  0) RUNS_ROOT="${PROJECT_ROOT}/results/runs/BambooQA_Phi_4_Reasoning_RG_CTO" ;;
  1) RUNS_ROOT="${PROJECT_ROOT}/results/runs/BambooQA_Phi_4_Reasoning_RG_CTO_gpu1" ;;
  2) RUNS_ROOT="${PROJECT_ROOT}/results/runs/BambooQA_Phi_4_Reasoning_RG_CTO_gpu2" ;;
  *)
    echo "ERROR: unsupported gpu_id=${GPU_ID} (expected 0, 1, or 2)"
    exit 1
    ;;
esac

export OUT_PREFIX="${RUNS_ROOT}/run0"

echo "========== RESUME BambooQA Phi-4 RG-CTO step5 | GPU${GPU_ID} =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Started at $(date '+%F %T')"

bash "${SCRIPT_DIR}/run_rg_cto_phi4_from_iter0_iter2.sh" 0 "$END_INDEX"

echo ""
echo "========== Resume done GPU${GPU_ID} at $(date '+%F %T') =========="
