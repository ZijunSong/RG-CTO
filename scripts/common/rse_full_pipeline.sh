#!/bin/bash
# Full RSE pipeline (Step 1-7) + Pass@1 evaluation for one run.
# Required env: MODEL_NAME, QUESTION_FILE, OUT_PREFIX
# Optional env: CUDA_VISIBLE_DEVICES, TENSOR_PARALLEL_SIZE, MAX_TOKENS, MAX_MODEL_LEN,
#               BATCH_SIZE, START_INDEX, END_INDEX, EMB_MODEL
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

START_INDEX="${1:-${START_INDEX:-0}}"
END_INDEX="${2:-${END_INDEX:-100}}"

: "${MODEL_NAME:?MODEL_NAME is required}"
: "${QUESTION_FILE:?QUESTION_FILE is required}"
: "${OUT_PREFIX:?OUT_PREFIX is required}"

export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2048}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export N_COMPLETIONS="${N_COMPLETIONS:-32}"
export MAX_TOKENS="${MAX_TOKENS:-38912}"
export N_EXP_COMPLETIONS="${N_EXP_COMPLETIONS:-32}"
export THRESHOLD="${THRESHOLD:-0.8}"
export EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

vllm_extra_args() {
  if [ -n "${MAX_MODEL_LEN:-}" ]; then
    printf -- '--max-model-len %s' "$MAX_MODEL_LEN"
  fi
}

VLLM_EXTRA="$(vllm_extra_args)"

echo "========== RSE Pipeline =========="
echo "  MODEL_NAME=${MODEL_NAME}"
echo "  QUESTION_FILE=${QUESTION_FILE}"
echo "  OUT_PREFIX=${OUT_PREFIX}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  TP=${TENSOR_PARALLEL_SIZE}  range=${START_INDEX}:${END_INDEX}"
echo "==================================="

rse_pass1_iter() {
  local iter="$1"
  local ver_dir="$2"
  if [ ! -d "$ver_dir" ]; then
    echo "  iter${iter}  N/A  (missing dir: $ver_dir)"
    return 0
  fi
  python eval/calculate_pass_at_k_from_completions.py \
    --verification_dir "$ver_dir" \
    --k_values 1 \
    --output_file "${ver_dir}/pass_at_1.json" \
    --max_reference 32 \
    --tokenizer_path "$MODEL_NAME" >/dev/null
  local PASS1
  PASS1="$(python - "$ver_dir" <<'PY'
import json, sys
ver_dir = sys.argv[1]
with open(ver_dir + "/pass_at_1.json", "r", encoding="utf-8") as f:
    m = json.load(f)
v = m.get("pass_at_k", {}).get("pass@1", None)
print("N/A" if v is None else f"{v*100:.2f}%")
PY
)"
  echo "  iter${iter}  ${PASS1}  (${ver_dir}/pass_at_1.json)"
}

# ---------- Step 1 ----------
echo "---------- Step 1: Baseline Sampling ----------"
mkdir -p "${OUT_PREFIX}_step1/results"
# shellcheck disable=SC2086
python code/standard_sampling.py \
  --model "$MODEL_NAME" \
  --input "$QUESTION_FILE" \
  --output "${OUT_PREFIX}_step1/results" \
  --n-completions "$N_COMPLETIONS" \
  --batch-size "$BATCH_SIZE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  $VLLM_EXTRA

# ---------- Step 2 ----------
echo "---------- Step 2: Experience Distillation ----------"
mkdir -p "${OUT_PREFIX}_step2/results"
# shellcheck disable=SC2086
python code/experience_distillation.py \
  --model "$MODEL_NAME" \
  --question-file "$QUESTION_FILE" \
  --answer-dir "${OUT_PREFIX}_step1/results" \
  --output-dir "${OUT_PREFIX}_step2/results" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --n-samples 1 \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  $VLLM_EXTRA

# ---------- Step 2.5 ----------
echo "---------- Step 2.5: Experience Deduplication ----------"
mkdir -p "${OUT_PREFIX}_step2/results_dedup" "${OUT_PREFIX}_step2/results_dedup_debug"
python code/experience_dedup.py \
  --experience-dir "${OUT_PREFIX}_step2/results" \
  --output-dir "${OUT_PREFIX}_step2/results_dedup" \
  --debug-dir "${OUT_PREFIX}_step2/results_dedup_debug" \
  --model-path "$EMB_MODEL" \
  --threshold "$THRESHOLD" \
  --keep-order

# ---------- Step 3 ----------
echo "---------- Step 3: Experience-Guided Search (iter1) ----------"
mkdir -p "${OUT_PREFIX}_step3/results"
# shellcheck disable=SC2086
python code/experience_guided_search.py \
  --model "$MODEL_NAME" \
  --input "$QUESTION_FILE" \
  --experience-dir "${OUT_PREFIX}_step2/results_dedup" \
  --output "${OUT_PREFIX}_step3/results" \
  --n-experience-completions "$N_EXP_COMPLETIONS" \
  --n-completions 32 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  $VLLM_EXTRA

# ---------- Step 4 ----------
echo "---------- Step 4: Experience Distillation ----------"
mkdir -p "${OUT_PREFIX}_step4/results"
# shellcheck disable=SC2086
python code/experience_distillation.py \
  --model "$MODEL_NAME" \
  --question-file "$QUESTION_FILE" \
  --answer-dir "${OUT_PREFIX}_step3/results" \
  --output-dir "${OUT_PREFIX}_step4/results" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --n-samples 1 \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  $VLLM_EXTRA

mkdir -p "${OUT_PREFIX}_step4/results_dedup" "${OUT_PREFIX}_step4/results_dedup_debug"
python code/experience_dedup.py \
  --experience-dir "${OUT_PREFIX}_step4/results" \
  --previous-experience-dir "${OUT_PREFIX}_step2/results_dedup" \
  --output-dir "${OUT_PREFIX}_step4/results_dedup" \
  --debug-dir "${OUT_PREFIX}_step4/results_dedup_debug" \
  --model-path "$EMB_MODEL" \
  --threshold "$THRESHOLD" \
  --keep-order

# ---------- Step 5 ----------
echo "---------- Step 5: Experience-Guided Search (iter2) ----------"
mkdir -p "${OUT_PREFIX}_step5/results"
# shellcheck disable=SC2086
python code/experience_guided_search.py \
  --model "$MODEL_NAME" \
  --input "$QUESTION_FILE" \
  --experience-dir "${OUT_PREFIX}_step4/results_dedup" \
  --output "${OUT_PREFIX}_step5/results" \
  --n-experience-completions "$N_EXP_COMPLETIONS" \
  --n-completions 32 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  $VLLM_EXTRA

# ---------- Step 6 ----------
echo "---------- Step 6: Experience Distillation ----------"
mkdir -p "${OUT_PREFIX}_step6/results"
# shellcheck disable=SC2086
python code/experience_distillation.py \
  --model "$MODEL_NAME" \
  --question-file "$QUESTION_FILE" \
  --answer-dir "${OUT_PREFIX}_step5/results" \
  --output-dir "${OUT_PREFIX}_step6/results" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --n-samples 1 \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  $VLLM_EXTRA

mkdir -p "${OUT_PREFIX}_step6/results_dedup" "${OUT_PREFIX}_step6/results_dedup_debug"
python code/experience_dedup.py \
  --experience-dir "${OUT_PREFIX}_step6/results" \
  --previous-experience-dir "${OUT_PREFIX}_step4/results_dedup" \
  --output-dir "${OUT_PREFIX}_step6/results_dedup" \
  --debug-dir "${OUT_PREFIX}_step6/results_dedup_debug" \
  --model-path "$EMB_MODEL" \
  --threshold "$THRESHOLD" \
  --keep-order

# ---------- Step 7 ----------
echo "---------- Step 7: Experience-Guided Search (iter3) ----------"
mkdir -p "${OUT_PREFIX}_step7/results"
# shellcheck disable=SC2086
python code/experience_guided_search.py \
  --model "$MODEL_NAME" \
  --input "$QUESTION_FILE" \
  --experience-dir "${OUT_PREFIX}_step6/results_dedup" \
  --output "${OUT_PREFIX}_step7/results" \
  --n-experience-completions "$N_EXP_COMPLETIONS" \
  --n-completions 32 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX" \
  $VLLM_EXTRA

# ---------- Evaluation ----------
echo "---------- Pass@1 (iter0..3) ----------"
declare -a ITER_DIRS=(
  "${OUT_PREFIX}_step1/results"
  "${OUT_PREFIX}_step3/results"
  "${OUT_PREFIX}_step5/results"
  "${OUT_PREFIX}_step7/results"
)
for iter in 0 1 2 3; do
  rse_pass1_iter "$iter" "${ITER_DIRS[$iter]}"
done

echo "========== RSE Pipeline done: ${OUT_PREFIX} =========="
