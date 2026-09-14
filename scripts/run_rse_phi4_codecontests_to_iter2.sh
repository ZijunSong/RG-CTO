#!/bin/bash
# RSE on CodeContests × Phi-4-Reasoning: iter0 (reuse if present) through iter2 (step1→step5).
# Required env: OUT_PREFIX
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

START_INDEX="${1:-0}"
END_INDEX="${2:-${END_INDEX:-165}}"

: "${OUT_PREFIX:?OUT_PREFIX is required}"

export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Phi-4-reasoning}"
export QUESTION_FILE="${QUESTION_FILE:-${PROJECT_ROOT}/data/CodeContests_Test_165.jsonl}"
export DATASET="${DATASET:-CodeContests}"

export BATCH_SIZE="${BATCH_SIZE:-2048}"
export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-32768}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export N_COMPLETIONS="${N_COMPLETIONS:-32}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"
export N_EXP_COMPLETIONS="${N_EXP_COMPLETIONS:-32}"
export THRESHOLD="${THRESHOLD:-0.85}"
export EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"
export EXPERIENCE_JUDGE_MODE="${EXPERIENCE_JUDGE_MODE:-llm_judge}"

export DISTILL_MAX_MODEL_LEN="${DISTILL_MAX_MODEL_LEN:-32768}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.88}"
export DISTILL_MAX_NUM_SEQS="${DISTILL_MAX_NUM_SEQS:-128}"
export EGS_MAX_MODEL_LEN="${EGS_MAX_MODEL_LEN:-32768}"
export EGS_GPU_MEMORY_UTILIZATION="${EGS_GPU_MEMORY_UTILIZATION:-0.88}"
export SAMPLE_GPU_MEMORY_UTILIZATION="${SAMPLE_GPU_MEMORY_UTILIZATION:-0.88}"

export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export TMPDIR="${TMPDIR:-/data/ppnm/tmp/rg_cto_rse_cc_phi4_$$}"
mkdir -p /data/ppnm/tmp "$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/ppnm/.cache}"
mkdir -p "$XDG_CACHE_HOME"

DATASET_ARGS=(--dataset "$DATASET")
_EXPECTED_RESULTS=$((END_INDEX - START_INDEX))

_has_step_results() {
  local dir="$1"
  compgen -G "${dir}/*" >/dev/null 2>&1
}

_step_results_count() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 -name '[0-9]*.json' 2>/dev/null | wc -l
}

_step_complete() {
  local dir="$1"
  local expect="$2"
  [ "$(_step_results_count "$dir")" -ge "$expect" ]
}

echo "========== RSE Phi-4 CodeContests iter0→iter2 | GPU=${CUDA_VISIBLE_DEVICES} TP=${TENSOR_PARALLEL_SIZE} =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "QUESTION_FILE=${QUESTION_FILE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "range=${START_INDEX}:${END_INDEX}"

echo "---------- Step 1: Baseline Sampling (iter0) ----------"
if _step_complete "${OUT_PREFIX}_step1/results" "$_EXPECTED_RESULTS"; then
  echo "[skip] step1 already done (${OUT_PREFIX}_step1/results)"
else
mkdir -p "${OUT_PREFIX}_step1/results"
python code/standard_sampling.py \
  --model "$MODEL_NAME" \
  --input "$QUESTION_FILE" \
  --output "${OUT_PREFIX}_step1/results" \
  --n-completions "$N_COMPLETIONS" \
  --batch-size "$BATCH_SIZE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$EGS_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$SAMPLE_GPU_MEMORY_UTILIZATION" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  "${DATASET_ARGS[@]}"
fi

echo "---------- Step 2: Experience Distillation ----------"
if _has_step_results "${OUT_PREFIX}_step2/results"; then
  echo "[skip] step2 already done (${OUT_PREFIX}_step2/results)"
else
mkdir -p "${OUT_PREFIX}_step2/results"
python code/experience_distillation.py \
  --model "$MODEL_NAME" \
  --question-file "$QUESTION_FILE" \
  --answer-dir "${OUT_PREFIX}_step1/results" \
  --output-dir "${OUT_PREFIX}_step2/results" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$DISTILL_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$DISTILL_GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$DISTILL_MAX_NUM_SEQS" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$DISTILL_MAX_TOKENS" \
  --n-samples 1 \
  --experience_judge_mode "$EXPERIENCE_JUDGE_MODE" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  "${DATASET_ARGS[@]}"
fi

echo "---------- Step 2.5: Experience Deduplication ----------"
if _has_step_results "${OUT_PREFIX}_step2/results_dedup"; then
  echo "[skip] step2.5 already done (${OUT_PREFIX}_step2/results_dedup)"
else
mkdir -p "${OUT_PREFIX}_step2/results_dedup" "${OUT_PREFIX}_step2/results_dedup_debug"
python code/experience_dedup.py \
  --experience-dir "${OUT_PREFIX}_step2/results" \
  --output-dir "${OUT_PREFIX}_step2/results_dedup" \
  --debug-dir "${OUT_PREFIX}_step2/results_dedup_debug" \
  --model-path "$EMB_MODEL" \
  --threshold "$THRESHOLD" \
  --keep-order
fi

echo "---------- Step 3: Experience-Guided Search (iter1) ----------"
if _step_complete "${OUT_PREFIX}_step3/results" "$_EXPECTED_RESULTS"; then
  echo "[skip] step3 already done (${OUT_PREFIX}_step3/results)"
else
mkdir -p "${OUT_PREFIX}_step3/results"
python code/experience_guided_search.py \
  --model "$MODEL_NAME" \
  --input "$QUESTION_FILE" \
  --experience-dir "${OUT_PREFIX}_step2/results_dedup" \
  --output "${OUT_PREFIX}_step3/results" \
  --n-experience-completions "$N_EXP_COMPLETIONS" \
  --n-completions 32 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$EGS_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$EGS_GPU_MEMORY_UTILIZATION" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  "${DATASET_ARGS[@]}"
fi

echo "---------- Step 4: Experience Distillation ----------"
if _has_step_results "${OUT_PREFIX}_step4/results" && _has_step_results "${OUT_PREFIX}_step4/results_dedup"; then
  echo "[skip] step4 already done (${OUT_PREFIX}_step4/results_dedup)"
else
mkdir -p "${OUT_PREFIX}_step4/results"
python code/experience_distillation.py \
  --model "$MODEL_NAME" \
  --question-file "$QUESTION_FILE" \
  --answer-dir "${OUT_PREFIX}_step3/results" \
  --output-dir "${OUT_PREFIX}_step4/results" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$DISTILL_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$DISTILL_GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$DISTILL_MAX_NUM_SEQS" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$DISTILL_MAX_TOKENS" \
  --n-samples 1 \
  --experience_judge_mode "$EXPERIENCE_JUDGE_MODE" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  "${DATASET_ARGS[@]}"

mkdir -p "${OUT_PREFIX}_step4/results_dedup" "${OUT_PREFIX}_step4/results_dedup_debug"
python code/experience_dedup.py \
  --experience-dir "${OUT_PREFIX}_step4/results" \
  --previous-experience-dir "${OUT_PREFIX}_step2/results_dedup" \
  --output-dir "${OUT_PREFIX}_step4/results_dedup" \
  --debug-dir "${OUT_PREFIX}_step4/results_dedup_debug" \
  --model-path "$EMB_MODEL" \
  --threshold "$THRESHOLD" \
  --keep-order
fi

echo "---------- Step 5: Experience-Guided Search (iter2) ----------"
if _step_complete "${OUT_PREFIX}_step5/results" "$_EXPECTED_RESULTS"; then
  echo "[skip] step5 already done (${OUT_PREFIX}_step5/results)"
else
mkdir -p "${OUT_PREFIX}_step5/results"
python code/experience_guided_search.py \
  --model "$MODEL_NAME" \
  --input "$QUESTION_FILE" \
  --experience-dir "${OUT_PREFIX}_step4/results_dedup" \
  --output "${OUT_PREFIX}_step5/results" \
  --n-experience-completions "$N_EXP_COMPLETIONS" \
  --n-completions 32 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$EGS_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$EGS_GPU_MEMORY_UTILIZATION" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  "${DATASET_ARGS[@]}"
fi

python - "$OUT_PREFIX" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
summary = {
    "dataset": "CodeContests",
    "model": "Phi-4-Reasoning",
    "method": "RSE",
    "note": "Code execution pass@k is not computed here; inspect *_step{1,3,5}/results for rollouts.",
    "steps": {
        "iter0": str(out) + "_step1/results",
        "iter1": str(out) + "_step3/results",
        "iter2": str(out) + "_step5/results",
    },
}
out_json = out.parent / f"{out.name}_eval_summary.json"
json.dump(summary, open(out_json, "w"), indent=2)
print(f"Wrote {out_json}")
PY

echo "========== Done: ${OUT_PREFIX} (iter0→iter2) =========="
