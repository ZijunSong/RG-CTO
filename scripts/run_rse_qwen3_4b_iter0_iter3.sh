#!/bin/bash
# Qwen3-4B-Thinking-2507 × RSE, iter0 through iter3 (step1–step7).
# Task prompts follow the question filename (BambooQA / CodeContests / TravelPlanner).
# Required env: CUDA_VISIBLE_DEVICES, QUESTION_FILE, OUT_PREFIX, END_INDEX
# Optional: DATASET, EVAL_TASK (qa|agent). CodeContests is scored with the code executor after the pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
: "${QUESTION_FILE:?Set QUESTION_FILE}"
: "${OUT_PREFIX:?Set OUT_PREFIX}"
: "${END_INDEX:?Set END_INDEX}"

export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Qwen3-4B-Thinking-2507}"
export TENSOR_PARALLEL_SIZE=1
export BATCH_SIZE="${BATCH_SIZE:-2048}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export N_COMPLETIONS="${N_COMPLETIONS:-32}"
export MAX_TOKENS="${MAX_TOKENS:-38912}"
export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-8192}"
export N_EXP_COMPLETIONS="${N_EXP_COMPLETIONS:-32}"
export THRESHOLD="${THRESHOLD:-0.85}"
export EXPERIENCE_JUDGE_MODE="${EXPERIENCE_JUDGE_MODE:-llm_judge}"
export EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"
export SEARCH_MAX_MODEL_LEN="${SEARCH_MAX_MODEL_LEN:-100000}"
export DISTILL_MAX_MODEL_LEN="${DISTILL_MAX_MODEL_LEN:-100000}"
export DISTILL_MAX_NUM_SEQS="${DISTILL_MAX_NUM_SEQS:-128}"
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-/data/ppnm/tmp/rse_qwen3_4b_gpu${CUDA_VISIBLE_DEVICES}_$$}"
mkdir -p /data/ppnm/tmp "$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/ppnm/.cache}"
mkdir -p "$XDG_CACHE_HOME"

GPU_UTIL="$(python - "$CUDA_VISIBLE_DEVICES" <<'PY'
import subprocess, sys
gpu = sys.argv[1].split(",")[0]
def q(field):
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    return float(out)
free, total = q("memory.free"), q("memory.total")
util = (free - 12288.0) / total
util = max(0.50, min(0.85, util))
print(f"{util:.2f}")
PY
)"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-$GPU_UTIL}"
export SEARCH_GPU_MEMORY_UTILIZATION="${SEARCH_GPU_MEMORY_UTILIZATION:-0.60}"

if [ -n "${EVAL_TASK:-}" ]; then
  export RGCTO_EVAL_TASK="$EVAL_TASK"
fi

echo "========== Qwen3-4B RSE iter0→iter3 | GPU=${CUDA_VISIBLE_DEVICES} =========="
echo "MODEL_NAME=${MODEL_NAME}"
echo "QUESTION_FILE=${QUESTION_FILE}"
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "DATASET=${DATASET:-inferred} EVAL_TASK=${EVAL_TASK:-none} range=0:${END_INDEX}"
echo "MAX_TOKENS=${MAX_TOKENS} DISTILL_MAX_TOKENS=${DISTILL_MAX_TOKENS}"
echo "search_util=${SEARCH_GPU_MEMORY_UTILIZATION} distill_util=${DISTILL_GPU_MEMORY_UTILIZATION}"
echo "Started at $(date '+%F %T')"

bash scripts/run_rse.sh 0 "$END_INDEX"

if [ "${DATASET:-}" = "CodeContests" ]; then
  echo "---------- CodeContests pass@1 ----------"
  for pair in "iter0:${OUT_PREFIX}_step1/results" "iter1:${OUT_PREFIX}_step3/results" "iter2:${OUT_PREFIX}_step5/results" "iter3:${OUT_PREFIX}_step7/results"; do
    label="${pair%%:*}"
    dir="${pair#*:}"
    CODE_EVAL_EXPECTED_N="$END_INDEX" \
      python scripts/eval_codecontests_pass1.py "$dir" "RSE_${label}" \
      || echo "WARNING: ${label} code pass@1 failed"
  done
fi

echo "========== Done: ${OUT_PREFIX} at $(date '+%F %T') =========="
