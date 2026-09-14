#!/bin/bash
# CTO (Phi-4-Reasoning): reuse frozen iter0, run iter1-2 (step2→step5).
# Required env: OUT_PREFIX, STEP1_RESULTS_SRC, QUESTION_FILE, DATASET
# Optional: CUDA_VISIBLE_DEVICES, END_INDEX, MAX_TOKENS override
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

START_INDEX="${1:-0}"
END_INDEX="${2:-${END_INDEX:-165}}"

: "${OUT_PREFIX:?OUT_PREFIX is required}"
: "${STEP1_RESULTS_SRC:?STEP1_RESULTS_SRC is required}"
: "${QUESTION_FILE:?QUESTION_FILE is required}"
: "${DATASET:?DATASET is required}"

export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Phi-4-reasoning}"

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
export RETRIEVAL_RERANK_MODEL="${RETRIEVAL_RERANK_MODEL:-/data/ppnm/models/cross-encoder-ms-marco-MiniLM-L-6-v2}"
export EXPERIENCE_JUDGE_MODE="${EXPERIENCE_JUDGE_MODE:-llm_judge}"

export ALPHA="${ALPHA:-0.7}"
export PLAUSIBILITY_TOP_K="${PLAUSIBILITY_TOP_K:-20}"
export CTO_AGG_MAX_PROP="${CTO_AGG_MAX_PROP:-96}"
export CTO_AGG_MAX_PIT="${CTO_AGG_MAX_PIT:-96}"

export DISTILL_MAX_MODEL_LEN="${DISTILL_MAX_MODEL_LEN:-32768}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"
export DISTILL_MAX_NUM_SEQS="${DISTILL_MAX_NUM_SEQS:-128}"
export CTO_MAX_MODEL_LEN="${CTO_MAX_MODEL_LEN:-32768}"
export CTO_GPU_MEMORY_UTILIZATION="${CTO_GPU_MEMORY_UTILIZATION:-0.70}"

export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-/data/ppnm/tmp/cto_phi4_gpu${CUDA_VISIBLE_DEVICES}_$$}"
mkdir -p /data/ppnm/tmp "$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/ppnm/.cache}"
mkdir -p "$XDG_CACHE_HOME"

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
  find "$dir" -maxdepth 1 \( -name '[0-9]*.json' -o -name '[0-9]*.jsonl' \) 2>/dev/null | wc -l
}

_step_complete() {
  local dir="$1"
  local expect="$2"
  [ "$(_step_results_count "$dir")" -ge "$expect" ]
}

_EXPECTED_RESULTS=$((END_INDEX - START_INDEX))

CTO_RETRIEVAL_ARGS=(
  --experience-retrieval embedding_rerank
  --retrieval-embedding-model "$EMB_MODEL"
  --retrieval-rerank-model "$RETRIEVAL_RERANK_MODEL"
  --retrieval-rerank-pool-mult "${CTO_RETRIEVAL_RERANK_POOL_MULT:-4}"
  --max-aggregated-propositions "${CTO_AGG_MAX_PROP}"
  --max-aggregated-pitfalls "${CTO_AGG_MAX_PIT}"
  --dataset "$DATASET"
)

echo "========== CTO Phi-4 iter0→iter2 | GPU=${CUDA_VISIBLE_DEVICES} | ${DATASET} =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "STEP1_RESULTS_SRC=${STEP1_RESULTS_SRC}"
echo "QUESTION_FILE=${QUESTION_FILE}"
echo "range=${START_INDEX}:${END_INDEX}"

echo "---------- Step 1: Reuse frozen iter0 (symlink) ----------"
if [ ! -d "$STEP1_RESULTS_SRC" ]; then
  echo "ERROR: STEP1_RESULTS_SRC missing: $STEP1_RESULTS_SRC"
  exit 1
fi
mkdir -p "${OUT_PREFIX}_step1"
ABS_SRC="$(cd "$STEP1_RESULTS_SRC" && pwd)"
ln -sfn "$ABS_SRC" "${OUT_PREFIX}_step1/results"
echo "Linked ${OUT_PREFIX}_step1/results -> $ABS_SRC"
echo "  iter0  $(_step_results_count "${OUT_PREFIX}_step1/results")/${_EXPECTED_RESULTS} rollouts"

echo "---------- Step 2: Experience Distillation (${EXPERIENCE_JUDGE_MODE}) ----------"
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
  --dataset "$DATASET" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX"
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

echo "---------- Step 3: CTO guided search (iter1) ----------"
if _step_complete "${OUT_PREFIX}_step3/results" "$_EXPECTED_RESULTS"; then
  echo "[skip] step3 already done (${OUT_PREFIX}_step3/results)"
else
mkdir -p "${OUT_PREFIX}_step3/results"
python code/cto_guided_search.py \
  --model "$MODEL_NAME" \
  --input "$QUESTION_FILE" \
  --experience-dir "${OUT_PREFIX}_step2/results_dedup" \
  --output "${OUT_PREFIX}_step3/results" \
  --n-experience-completions "$N_EXP_COMPLETIONS" \
  --n-completions "$N_COMPLETIONS" \
  --alpha "$ALPHA" \
  --plausibility-top-k "$PLAUSIBILITY_TOP_K" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$CTO_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$CTO_GPU_MEMORY_UTILIZATION" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  "${CTO_RETRIEVAL_ARGS[@]}" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX"
fi
echo "  iter1  $(_step_results_count "${OUT_PREFIX}_step3/results")/${_EXPECTED_RESULTS} rollouts"

echo "---------- Step 4: Experience Distillation (${EXPERIENCE_JUDGE_MODE}) ----------"
if _has_step_results "${OUT_PREFIX}_step4/results" && _step_complete "${OUT_PREFIX}_step4/results_dedup" "$_EXPECTED_RESULTS"; then
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
  --dataset "$DATASET" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX"

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

echo "---------- Step 5: CTO guided search (iter2) ----------"
if _step_complete "${OUT_PREFIX}_step5/results" "$_EXPECTED_RESULTS"; then
  echo "[skip] step5 already done (${OUT_PREFIX}_step5/results)"
else
mkdir -p "${OUT_PREFIX}_step5/results"
python code/cto_guided_search.py \
  --model "$MODEL_NAME" \
  --input "$QUESTION_FILE" \
  --experience-dir "${OUT_PREFIX}_step4/results_dedup" \
  --output "${OUT_PREFIX}_step5/results" \
  --n-experience-completions "$N_EXP_COMPLETIONS" \
  --n-completions "$N_COMPLETIONS" \
  --alpha "$ALPHA" \
  --plausibility-top-k "$PLAUSIBILITY_TOP_K" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$CTO_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$CTO_GPU_MEMORY_UTILIZATION" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  "${CTO_RETRIEVAL_ARGS[@]}" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX"
fi
echo "  iter2  $(_step_results_count "${OUT_PREFIX}_step5/results")/${_EXPECTED_RESULTS} rollouts"

python - "$OUT_PREFIX" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
summary = {
    "dataset": "CodeContests",
    "model": "Phi-4-Reasoning",
    "method": "CTO",
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

echo "========== Done: ${OUT_PREFIX} =========="
