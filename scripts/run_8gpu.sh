#!/bin/bash
# Eight-GPU data-parallel entry.
# Each GPU runs its own single-card vLLM engine on a disjoint question shard.
# A step starts only after every shard of the previous step has finished and
# the per-question files have been gathered into the shared step directory.
#
# Usage:
#   export MODEL_NAME=/path/to/model
#   export QUESTION_FILE=data/TravelPlanner_Val60.jsonl
#   export OUT_PREFIX=runs/TravelPlanner_Val60_RG_CTO
#   export METHOD=rg_cto          # rg_cto | cto | rse
#   export DATASET=TravelPlanner_Val60   # optional; inferred from the filename if unset
#   bash scripts/run_8gpu.sh 0 60
#
# If the scheduler already set CUDA_VISIBLE_DEVICES to the 8 allocated ids,
# those ids are used. Otherwise GPUs 0-7 are used.
# NGPU overrides the worker count (default 8).
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/run_8gpu.sh <start_index> <end_index>"
  exit 1
fi

START_INDEX=$1
END_INDEX=$2
if [ "$END_INDEX" -le "$START_INDEX" ]; then
  echo "end_index must be greater than start_index"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${PYTHON:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  else
    PYTHON=python3
  fi
fi

METHOD="${METHOD:-rg_cto}"
NGPU="${NGPU:-8}"

: "${MODEL_NAME:?Set MODEL_NAME}"
: "${QUESTION_FILE:?Set QUESTION_FILE}"
: "${OUT_PREFIX:?Set OUT_PREFIX}"

export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export N_COMPLETIONS="${N_COMPLETIONS:-32}"
export MAX_TOKENS="${MAX_TOKENS:-38912}"
export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-8192}"
export BATCH_SIZE="${BATCH_SIZE:-2048}"
export THRESHOLD="${THRESHOLD:-0.85}"
export EXPERIENCE_JUDGE_MODE="${EXPERIENCE_JUDGE_MODE:-llm_judge}"
export EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"
export RETRIEVAL_RERANK_MODEL="${RETRIEVAL_RERANK_MODEL:-/data/ppnm/models/cross-encoder-ms-marco-MiniLM-L-6-v2}"
export DISTILL_MAX_MODEL_LEN="${DISTILL_MAX_MODEL_LEN:-100000}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"
export DISTILL_MAX_NUM_SEQS="${DISTILL_MAX_NUM_SEQS:-128}"
export CTO_AGG_MAX_PROP="${CTO_AGG_MAX_PROP:-96}"
export CTO_AGG_MAX_PIT="${CTO_AGG_MAX_PIT:-96}"
export CTO_RETRIEVAL_RERANK_POOL_MULT="${CTO_RETRIEVAL_RERANK_POOL_MULT:-8}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

case "$METHOD" in
  rg_cto)
    export ALPHA="${ALPHA:-0.7}"
    export GATE_DELTA="${GATE_DELTA:-0.4}"
    export TAU_MATCH="${TAU_MATCH:-0.8}"
    export LAMBDA_U="${LAMBDA_U:-0.5}"
    export LAMBDA_L="${LAMBDA_L:-0.5}"
    export PILOT_N="${PILOT_N:-4}"
    export MIN_PITFALL_SUPPORT="${MIN_PITFALL_SUPPORT:-2}"
    export N_EXPERIENCE_COMPLETIONS="${N_EXPERIENCE_COMPLETIONS:-48}"
    SEARCH_MAX_MODEL_LEN="${RGCTO_MAX_MODEL_LEN:-100000}"
    SEARCH_GPU_MEMORY_UTILIZATION="${RGCTO_GPU_MEMORY_UTILIZATION:-0.60}"
    ;;
  cto)
    export ALPHA="${ALPHA:-0.55}"
    export PLAUSIBILITY_TOP_K="${PLAUSIBILITY_TOP_K:-8}"
    export N_EXPERIENCE_COMPLETIONS="${N_EXPERIENCE_COMPLETIONS:-48}"
    SEARCH_MAX_MODEL_LEN="${CTO_MAX_MODEL_LEN:-100000}"
    SEARCH_GPU_MEMORY_UTILIZATION="${CTO_GPU_MEMORY_UTILIZATION:-0.60}"
    ;;
  rse)
    export N_EXPERIENCE_COMPLETIONS="${N_EXPERIENCE_COMPLETIONS:-32}"
    SEARCH_MAX_MODEL_LEN="${SEARCH_MAX_MODEL_LEN:-100000}"
    SEARCH_GPU_MEMORY_UTILIZATION="${SEARCH_GPU_MEMORY_UTILIZATION:-0.60}"
    ;;
  *)
    echo "METHOD must be rg_cto, cto, or rse (got ${METHOD})"
    exit 1
    ;;
esac
export MAX_PILOT_TOKENS="${MAX_PILOT_TOKENS:-$MAX_TOKENS}"

TASK_TYPE="$(
  "$PYTHON" - "${DATASET:-}" "$QUESTION_FILE" <<'PY'
import sys
sys.path.insert(0, "code")
from task_prompts import resolve_task_type
dataset = sys.argv[1] or None
print(resolve_task_type(dataset=dataset, input_path=sys.argv[2]))
PY
)"
case "$TASK_TYPE" in
  math|qa|code|agent) ;;
  *)
    echo "Unsupported task type: ${TASK_TYPE}"
    exit 1
    ;;
esac

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a ALL_GPUS <<< "${CUDA_VISIBLE_DEVICES// /}"
else
  ALL_GPUS=()
  for ((i = 0; i < NGPU; i++)); do
    ALL_GPUS+=("$i")
  done
fi
if [ "${#ALL_GPUS[@]}" -lt "$NGPU" ]; then
  echo "Need ${NGPU} GPUs, CUDA_VISIBLE_DEVICES provides ${#ALL_GPUS[@]}: ${ALL_GPUS[*]:-none}"
  exit 1
fi
WORKER_GPUS=("${ALL_GPUS[@]:0:NGPU}")

TOTAL=$((END_INDEX - START_INDEX))
BASE=$((TOTAL / NGPU))
REM=$((TOTAL % NGPU))
SHARD_START=()
SHARD_END=()
CURSOR=$START_INDEX
for ((i = 0; i < NGPU; i++)); do
  SPAN=$BASE
  if [ "$i" -lt "$REM" ]; then
    SPAN=$((BASE + 1))
  fi
  SHARD_START+=("$CURSOR")
  CURSOR=$((CURSOR + SPAN))
  SHARD_END+=("$CURSOR")
done

LOG_DIR="${OUT_PREFIX}_logs"

DATASET_ARGS=()
if [ -n "${DATASET:-}" ]; then
  DATASET_ARGS=(--dataset "$DATASET")
fi

count_indexed() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 -type f \( -name '[0-9]*.json' -o -name '[0-9]*.jsonl' \) | wc -l | tr -d ' '
}

echo "========== 8-GPU ${METHOD} =========="
echo "MODEL_NAME=${MODEL_NAME}"
echo "QUESTION_FILE=${QUESTION_FILE}"
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "task_type=${TASK_TYPE}  dataset=${DATASET:-inferred}"
echo "range=${START_INDEX}:${END_INDEX}  (${TOTAL} questions, ${NGPU} shards)"
echo "GPUs=${WORKER_GPUS[*]}"
for ((i = 0; i < NGPU; i++)); do
  echo "  shard ${i}: GPU ${WORKER_GPUS[$i]}  [${SHARD_START[$i]}, ${SHARD_END[$i]})"
done
echo "====================================="

if [ "${PLAN_ONLY:-0}" = 1 ]; then
  exit 0
fi

mkdir -p "$LOG_DIR"
for step in 1 2 3 4 5 6 7; do
  mkdir -p "${OUT_PREFIX}_step${step}/results"
done
for step in 2 4 6; do
  mkdir -p "${OUT_PREFIX}_step${step}/results_dedup" "${OUT_PREFIX}_step${step}/results_dedup_debug"
done

trap 'jobs -p | xargs -r kill 2>/dev/null || true' INT TERM

run_vllm_step() {
  local name="$1"
  local out_dir="$2"
  shift 2
  local pids=()
  local i s e gpu log
  echo ""
  echo "---------- ${name} ----------"
  for ((i = 0; i < NGPU; i++)); do
    s=${SHARD_START[$i]}
    e=${SHARD_END[$i]}
    gpu=${WORKER_GPUS[$i]}
    if [ "$s" -ge "$e" ]; then
      echo "  GPU ${gpu} idle (empty shard)"
      continue
    fi
    log="${LOG_DIR}/${name}_gpu${gpu}_${s}_${e}.log"
    echo "  GPU ${gpu}  [${s}, ${e})  -> ${log}"
    CUDA_VISIBLE_DEVICES="$gpu" \
      TENSOR_PARALLEL_SIZE=1 \
      NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
      NCCL_NVLS_ENABLE="$NCCL_NVLS_ENABLE" \
      "$PYTHON" "$@" --tensor-parallel-size 1 --start-idx "$s" --end-idx "$e" \
      >"$log" 2>&1 &
    pids+=("$!")
  done
  local fail=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [ "$fail" -ne 0 ]; then
    echo "ERROR: ${name} failed. See ${LOG_DIR}/${name}_gpu*.log"
    exit 1
  fi
  local n
  n="$(count_indexed "$out_dir")"
  echo "  gathered ${n} / ${TOTAL} result files in ${out_dir}"
  if [ "$n" -lt "$TOTAL" ]; then
    echo "ERROR: ${name} produced ${n} files, expected ${TOTAL}"
    exit 1
  fi
}

run_dedup() {
  local round="$1"
  local prev_dedup="${2:-}"
  local exp_dir="${OUT_PREFIX}_step${round}"
  echo ""
  echo "---------- Step ${round}.5: dedup (all shards) ----------"
  local -a args=(
    code/experience_dedup.py
    --experience-dir "${exp_dir}/results"
    --output-dir "${exp_dir}/results_dedup"
    --debug-dir "${exp_dir}/results_dedup_debug"
    --model-path "$EMB_MODEL"
    --threshold "$THRESHOLD"
    --keep-order
    --workers "$NGPU"
  )
  if [ -n "$prev_dedup" ]; then
    args+=(--previous-experience-dir "$prev_dedup")
  fi
  local log="${LOG_DIR}/dedup_step${round}.log"
  CUDA_VISIBLE_DEVICES="${WORKER_GPUS[0]}" "$PYTHON" "${args[@]}" >"$log" 2>&1
  local n
  n="$(count_indexed "${exp_dir}/results_dedup")"
  echo "  gathered ${n} / ${TOTAL} dedup files in ${exp_dir}/results_dedup"
  if [ "$n" -lt "$TOTAL" ]; then
    echo "ERROR: dedup step ${round} produced ${n} files, expected ${TOTAL}. See ${log}"
    exit 1
  fi
}

maybe_pass1() {
  local iter="$1"
  local dir="$2"
  if [ "$TASK_TYPE" = "code" ]; then
    if ! "$PYTHON" scripts/eval_codecontests_pass1.py "$dir" "iter${iter}" \
      >"${LOG_DIR}/pass1_iter${iter}.log" 2>&1; then
      echo "  iter${iter} code pass@1 skipped (see ${LOG_DIR}/pass1_iter${iter}.log)"
      return 0
    fi
    "$PYTHON" - "$dir" <<'PY'
import json, sys
path = sys.argv[1] + "/code_pass_at_1.json"
with open(path, encoding="utf-8") as f:
    metrics = json.load(f)
print(f"  iter code pass@1 {metrics.get('pass_at_1_pct')}%  ({path})")
PY
    return 0
  fi
  if ! RGCTO_EVAL_TASK="$TASK_TYPE" "$PYTHON" eval/calculate_pass_at_k_from_completions.py \
    --verification_dir "$dir" \
    --k_values 1 \
    --output_file "${dir}/pass_at_1.json" \
    --max_reference 32 \
    --tokenizer_path "$MODEL_NAME" >"${LOG_DIR}/pass1_iter${iter}.log" 2>&1; then
    echo "  iter${iter} pass@1 skipped (see ${LOG_DIR}/pass1_iter${iter}.log)"
    return 0
  fi
  "$PYTHON" - "$dir" <<'PY'
import json, sys
path = sys.argv[1] + "/pass_at_1.json"
with open(path, encoding="utf-8") as f:
    metrics = json.load(f)
value = metrics.get("pass_at_k", {}).get("pass@1")
extra = metrics.get("travelplanner") or {}
line = "N/A" if value is None else f"{value * 100:.2f}%"
print(f"  iter pass@1 {line}  ({path})")
if extra:
    print(
        "  delivery={delivery_rate:.2%}  commonsense micro={commonsense_micro:.2%}  "
        "hard micro={hard_micro:.2%}".format(**extra)
    )
PY
}

build_sampling_args() {
  SAMPLING_ARGS=(
    code/standard_sampling.py
    --model "$MODEL_NAME"
    --input "$QUESTION_FILE"
    --output "${OUT_PREFIX}_step1/results"
    --n-completions "$N_COMPLETIONS"
    --batch-size "$BATCH_SIZE"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$SEARCH_MAX_MODEL_LEN"
    --gpu-memory-utilization "$SEARCH_GPU_MEMORY_UTILIZATION"
  )
  SAMPLING_ARGS+=(--task-type "$TASK_TYPE")
  if [ -n "${DATASET:-}" ]; then
    SAMPLING_ARGS+=(--dataset "$DATASET")
  fi
}

build_distill_args() {
  local answer_dir="$1"
  local out_dir="$2"
  DISTILL_ARGS=(
    code/experience_distillation.py
    --model "$MODEL_NAME"
    --question-file "$QUESTION_FILE"
    --answer-dir "$answer_dir"
    --output-dir "$out_dir"
    --max-model-len "$DISTILL_MAX_MODEL_LEN"
    --gpu-memory-utilization "$DISTILL_GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$DISTILL_MAX_NUM_SEQS"
    --batch-size "$BATCH_SIZE"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --max-tokens "$DISTILL_MAX_TOKENS"
    --n-samples 1
    --experience_judge_mode "$EXPERIENCE_JUDGE_MODE"
  )
  DISTILL_ARGS+=(--task-type "$TASK_TYPE")
  if [ -n "${DATASET:-}" ]; then
    DISTILL_ARGS+=(--dataset "$DATASET")
  fi
}

build_guided_args() {
  local exp_dir="$1"
  local answer_dir="$2"
  local out_dir="$3"
  case "$METHOD" in
    rg_cto)
      GUIDED_ARGS=(
        code/rg_cto_guided_search.py
        --model "$MODEL_NAME"
        --input "$QUESTION_FILE"
        --experience-dir "$exp_dir"
        --answer-dir "$answer_dir"
        --output "$out_dir"
        --n-experience-completions "$N_EXPERIENCE_COMPLETIONS"
        --n-completions "$N_COMPLETIONS"
        --alpha "$ALPHA"
        --gate-delta "$GATE_DELTA"
        --tau-match "$TAU_MATCH"
        --lambda-u "$LAMBDA_U"
        --lambda-l "$LAMBDA_L"
        --pilot-n "$PILOT_N"
        --min-pitfall-support "$MIN_PITFALL_SUPPORT"
        --max-pilot-tokens "$MAX_PILOT_TOKENS"
        --max-model-len "$SEARCH_MAX_MODEL_LEN"
        --gpu-memory-utilization "$SEARCH_GPU_MEMORY_UTILIZATION"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --top-k "$TOP_K"
        --max-tokens "$MAX_TOKENS"
        --experience-retrieval embedding_rerank
        --retrieval-embedding-model "$EMB_MODEL"
        --retrieval-rerank-model "$RETRIEVAL_RERANK_MODEL"
        --retrieval-rerank-pool-mult "$CTO_RETRIEVAL_RERANK_POOL_MULT"
        --max-aggregated-propositions "$CTO_AGG_MAX_PROP"
        --max-aggregated-pitfalls "$CTO_AGG_MAX_PIT"
      )
      ;;
    cto)
      GUIDED_ARGS=(
        code/cto_guided_search.py
        --model "$MODEL_NAME"
        --input "$QUESTION_FILE"
        --experience-dir "$exp_dir"
        --output "$out_dir"
        --n-experience-completions "$N_EXPERIENCE_COMPLETIONS"
        --n-completions "$N_COMPLETIONS"
        --alpha "$ALPHA"
        --plausibility-top-k "$PLAUSIBILITY_TOP_K"
        --max-model-len "$SEARCH_MAX_MODEL_LEN"
        --gpu-memory-utilization "$SEARCH_GPU_MEMORY_UTILIZATION"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --top-k "$TOP_K"
        --max-tokens "$MAX_TOKENS"
        --experience-retrieval embedding_rerank
        --retrieval-embedding-model "$EMB_MODEL"
        --retrieval-rerank-model "$RETRIEVAL_RERANK_MODEL"
        --retrieval-rerank-pool-mult "$CTO_RETRIEVAL_RERANK_POOL_MULT"
        --max-aggregated-propositions "$CTO_AGG_MAX_PROP"
        --max-aggregated-pitfalls "$CTO_AGG_MAX_PIT"
      )
      ;;
    rse)
      GUIDED_ARGS=(
        code/experience_guided_search.py
        --model "$MODEL_NAME"
        --input "$QUESTION_FILE"
        --experience-dir "$exp_dir"
        --output "$out_dir"
        --n-experience-completions "$N_EXPERIENCE_COMPLETIONS"
        --n-completions "$N_COMPLETIONS"
        --batch-size "$BATCH_SIZE"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --top-k "$TOP_K"
        --max-tokens "$MAX_TOKENS"
        --max-model-len "$SEARCH_MAX_MODEL_LEN"
        --gpu-memory-utilization "$SEARCH_GPU_MEMORY_UTILIZATION"
      )
      ;;
  esac
  GUIDED_ARGS+=(--task-type "$TASK_TYPE")
  if [ -n "${DATASET:-}" ]; then
    GUIDED_ARGS+=(--dataset "$DATASET")
  fi
}

build_sampling_args
run_vllm_step "step1_sample" "${OUT_PREFIX}_step1/results" "${SAMPLING_ARGS[@]}"
maybe_pass1 0 "${OUT_PREFIX}_step1/results"

build_distill_args "${OUT_PREFIX}_step1/results" "${OUT_PREFIX}_step2/results"
run_vllm_step "step2_distill" "${OUT_PREFIX}_step2/results" "${DISTILL_ARGS[@]}"
run_dedup 2

build_guided_args "${OUT_PREFIX}_step2/results_dedup" "${OUT_PREFIX}_step1/results" "${OUT_PREFIX}_step3/results"
run_vllm_step "step3_guided" "${OUT_PREFIX}_step3/results" "${GUIDED_ARGS[@]}"
maybe_pass1 1 "${OUT_PREFIX}_step3/results"

build_distill_args "${OUT_PREFIX}_step3/results" "${OUT_PREFIX}_step4/results"
run_vllm_step "step4_distill" "${OUT_PREFIX}_step4/results" "${DISTILL_ARGS[@]}"
run_dedup 4 "${OUT_PREFIX}_step2/results_dedup"

build_guided_args "${OUT_PREFIX}_step4/results_dedup" "${OUT_PREFIX}_step3/results" "${OUT_PREFIX}_step5/results"
run_vllm_step "step5_guided" "${OUT_PREFIX}_step5/results" "${GUIDED_ARGS[@]}"
maybe_pass1 2 "${OUT_PREFIX}_step5/results"

build_distill_args "${OUT_PREFIX}_step5/results" "${OUT_PREFIX}_step6/results"
run_vllm_step "step6_distill" "${OUT_PREFIX}_step6/results" "${DISTILL_ARGS[@]}"
run_dedup 6 "${OUT_PREFIX}_step4/results_dedup"

build_guided_args "${OUT_PREFIX}_step6/results_dedup" "${OUT_PREFIX}_step5/results" "${OUT_PREFIX}_step7/results"
run_vllm_step "step7_guided" "${OUT_PREFIX}_step7/results" "${GUIDED_ARGS[@]}"
maybe_pass1 3 "${OUT_PREFIX}_step7/results"

echo ""
echo "8-GPU ${METHOD} finished. Final results: ${OUT_PREFIX}_step7/results"
