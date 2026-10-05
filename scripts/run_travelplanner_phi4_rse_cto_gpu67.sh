#!/bin/bash
# TravelPlanner Val60 × Phi-4-Reasoning.
# GPU 6 and GPU 7 first split iter0 sampling into one shared rollout dir.
# After that finishes, GPU 6 runs RSE iter1-iter3 and GPU 7 runs CTO iter1-iter3,
# both reading the same iter0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Phi-4-reasoning}"
export QUESTION_FILE="${QUESTION_FILE:-${PROJECT_ROOT}/data/TravelPlanner_Val60.jsonl}"
export DATASET="${DATASET:-TravelPlanner_Val60}"
export EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"
export RETRIEVAL_RERANK_MODEL="${RETRIEVAL_RERANK_MODEL:-/data/ppnm/models/cross-encoder-ms-marco-MiniLM-L-6-v2}"

export BATCH_SIZE="${BATCH_SIZE:-2048}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export N_COMPLETIONS="${N_COMPLETIONS:-32}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"
export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-32768}"
export THRESHOLD="${THRESHOLD:-0.85}"
export EXPERIENCE_JUDGE_MODE="${EXPERIENCE_JUDGE_MODE:-llm_judge}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export DISTILL_MAX_NUM_SEQS="${DISTILL_MAX_NUM_SEQS:-32}"

export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/ppnm/.cache}"
mkdir -p /data/ppnm/tmp "$XDG_CACHE_HOME"

GPU_RSE="${GPU_RSE:-6}"
GPU_CTO="${GPU_CTO:-7}"
END_INDEX="${END_INDEX:-60}"
MID_INDEX="${MID_INDEX:-30}"

RUNS_ROOT="${PROJECT_ROOT}/results/runs"
RUN_TAG="${RUN_TAG:-}"
TAG_SUFFIX="${RUN_TAG:+_${RUN_TAG}}"
SHARED_PREFIX="${RUNS_ROOT}/TravelPlanner_Val60_Phi_4_Reasoning_shared${TAG_SUFFIX}/run0"
RSE_PREFIX="${RUNS_ROOT}/TravelPlanner_Val60_Phi_4_Reasoning_RSE${TAG_SUFFIX}/run0"
CTO_PREFIX="${RUNS_ROOT}/TravelPlanner_Val60_Phi_4_Reasoning_CTO${TAG_SUFFIX}/run0"
LOG_DIR="${PROJECT_ROOT}/logs/travelplanner_phi4_gpu67${TAG_SUFFIX}"
mkdir -p "$LOG_DIR" \
  "${SHARED_PREFIX}_step1/results" \
  "$(dirname "$RSE_PREFIX")" \
  "$(dirname "$CTO_PREFIX")"

_count() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 \( -name '[0-9]*.json' -o -name '[0-9]*.jsonl' \) 2>/dev/null | wc -l
}

_gpu_util() {
  local gpu="$1"
  python - "$gpu" <<'PY'
import subprocess, sys
gpu = sys.argv[1]
def q(field):
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    return float(out)
free, total = q("memory.free"), q("memory.total")
# vLLM budgets total*util, while weights come out of the currently free pool.
util = (free - 12288.0) / total
util = max(0.55, min(0.82, util))
print(f"{util:.2f}")
PY
}

_score_agent() {
  local iter="$1"
  local dir="$2"
  echo "---------- score iter${iter}: ${dir} ----------"
  if ! RGCTO_EVAL_TASK=agent python eval/calculate_pass_at_k_from_completions.py \
      --verification_dir "$dir" \
      --k_values 1 \
      --output_file "${dir}/pass_at_1.json" \
      --max_reference 32 \
      --tokenizer_path "$MODEL_NAME" \
      >"${LOG_DIR}/pass1_iter${iter}_$(basename "$(dirname "$dir")").log" 2>&1; then
    echo "WARNING: iter${iter} pass@1 failed (see ${LOG_DIR})"
    return 0
  fi
  python - "${dir}/pass_at_1.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1], encoding="utf-8"))
value = metrics.get("pass_at_k", {}).get("pass@1")
extra = metrics.get("travelplanner") or {}
line = "N/A" if value is None else f"{value * 100:.2f}%"
print(f"  pass@1 {line}  ({sys.argv[1]})")
if extra:
    print(
        "  delivery={delivery_rate:.2%}  commonsense micro={commonsense_micro:.2%}  "
        "hard micro={hard_micro:.2%}".format(**extra)
    )
PY
}

_sample_shard() {
  local gpu="$1"
  local start="$2"
  local end="$3"
  local util
  util="$(_gpu_util "$gpu")"
  echo "iter0 shard GPU ${gpu} [${start}, ${end}) util=${util}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  TMPDIR="/data/ppnm/tmp/tp_phi4_iter0_gpu${gpu}_$$" \
  python code/standard_sampling.py \
    --model "$MODEL_NAME" \
    --input "$QUESTION_FILE" \
    --output "${SHARED_PREFIX}_step1/results" \
    --n-completions "$N_COMPLETIONS" \
    --batch-size "$BATCH_SIZE" \
    --tensor-parallel-size 1 \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$util" \
    --dataset "$DATASET" \
    --task-type agent \
    --start-idx "$start" \
    --end-idx "$end"
}

_link_iter0() {
  local prefix="$1"
  mkdir -p "${prefix}_step1"
  ln -sfn "$(cd "${SHARED_PREFIX}_step1/results" && pwd)" "${prefix}_step1/results"
  echo "Linked ${prefix}_step1/results -> ${SHARED_PREFIX}_step1/results ($(_count "${prefix}_step1/results") files)"
}

_distill_round() {
  local gpu="$1"
  local util="$2"
  local round="$3"
  local answer_dir="$4"
  local prev_dedup="${5:-}"
  local prefix="$6"
  local exp="${prefix}_step${round}"
  if [ "$(_count "${exp}/results")" -ge "$END_INDEX" ] && [ "$(_count "${exp}/results_dedup")" -ge "$END_INDEX" ]; then
    echo "[skip] step${round} already done ($(_count "${exp}/results_dedup")/${END_INDEX})"
    return 0
  fi
  if [ "$(_count "${exp}/results")" -ge "$END_INDEX" ]; then
    echo "[skip] step${round} distillation already done; running dedup only"
  else
  mkdir -p "${exp}/results"
  CUDA_VISIBLE_DEVICES="$gpu" \
  TMPDIR="/data/ppnm/tmp/tp_phi4_distill_gpu${gpu}_step${round}_$$" \
  python code/experience_distillation.py \
    --model "$MODEL_NAME" \
    --question-file "$QUESTION_FILE" \
    --answer-dir "$answer_dir" \
    --output-dir "${exp}/results" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$util" \
    --max-num-seqs "$DISTILL_MAX_NUM_SEQS" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$DISTILL_MAX_TOKENS" \
    --n-samples 1 \
    --experience_judge_mode "$EXPERIENCE_JUDGE_MODE" \
    --dataset "$DATASET" \
    --task-type agent \
    --start-idx 0 \
    --end-idx "$END_INDEX"
  fi
  mkdir -p "${exp}/results_dedup" "${exp}/results_dedup_debug"
  local -a dedup_args=(
    code/experience_dedup.py
    --experience-dir "${exp}/results"
    --output-dir "${exp}/results_dedup"
    --debug-dir "${exp}/results_dedup_debug"
    --model-path "$EMB_MODEL"
    --threshold "$THRESHOLD"
    --keep-order
  )
  if [ -n "$prev_dedup" ]; then
    dedup_args+=(--previous-experience-dir "$prev_dedup")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python "${dedup_args[@]}"
}

_guided_rse() {
  local gpu="$1"
  local util="$2"
  local exp_dir="$3"
  local out_dir="$4"
  if [ "$(_count "$out_dir")" -ge "$END_INDEX" ]; then
    echo "[skip] RSE guided already done ($(_count "$out_dir")/${END_INDEX}) ${out_dir}"
    return 0
  fi
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" \
  TMPDIR="/data/ppnm/tmp/tp_phi4_rse_gpu${gpu}_$$" \
  python code/experience_guided_search.py \
    --model "$MODEL_NAME" \
    --input "$QUESTION_FILE" \
    --experience-dir "$exp_dir" \
    --output "$out_dir" \
    --n-experience-completions 32 \
    --n-completions "$N_COMPLETIONS" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$util" \
    --tensor-parallel-size 1 \
    --dataset "$DATASET" \
    --task-type agent \
    --start-idx 0 \
    --end-idx "$END_INDEX"
}

_guided_cto() {
  local gpu="$1"
  local util="$2"
  local exp_dir="$3"
  local out_dir="$4"
  if [ "$(_count "$out_dir")" -ge "$END_INDEX" ]; then
    echo "[skip] CTO guided already done ($(_count "$out_dir")/${END_INDEX}) ${out_dir}"
    return 0
  fi
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" \
  TMPDIR="/data/ppnm/tmp/tp_phi4_cto_gpu${gpu}_$$" \
  python code/cto_guided_search.py \
    --model "$MODEL_NAME" \
    --input "$QUESTION_FILE" \
    --experience-dir "$exp_dir" \
    --output "$out_dir" \
    --n-experience-completions 48 \
    --n-completions "$N_COMPLETIONS" \
    --alpha 0.55 \
    --plausibility-top-k 8 \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$util" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --experience-retrieval embedding_rerank \
    --retrieval-embedding-model "$EMB_MODEL" \
    --retrieval-rerank-model "$RETRIEVAL_RERANK_MODEL" \
    --retrieval-rerank-pool-mult 8 \
    --max-aggregated-propositions 96 \
    --max-aggregated-pitfalls 96 \
    --dataset "$DATASET" \
    --task-type agent \
    --start-idx 0 \
    --end-idx "$END_INDEX"
}

_run_method() {
  local method="$1"
  local gpu="$2"
  local prefix="$3"
  local util
  util="$(_gpu_util "$gpu")"
  echo "========== ${method} iter1-iter3 | GPU=${gpu} util=${util} | ${prefix} =========="
  _link_iter0 "$prefix"

  _distill_round "$gpu" "$util" 2 "${prefix}_step1/results" "" "$prefix"
  if [ "$method" = "rse" ]; then
    _guided_rse "$gpu" "$util" "${prefix}_step2/results_dedup" "${prefix}_step3/results"
  else
    _guided_cto "$gpu" "$util" "${prefix}_step2/results_dedup" "${prefix}_step3/results"
  fi
  echo "  iter1 $(_count "${prefix}_step3/results")/${END_INDEX}"
  _score_agent 1 "${prefix}_step3/results"

  _distill_round "$gpu" "$util" 4 "${prefix}_step3/results" "${prefix}_step2/results_dedup" "$prefix"
  if [ "$method" = "rse" ]; then
    _guided_rse "$gpu" "$util" "${prefix}_step4/results_dedup" "${prefix}_step5/results"
  else
    _guided_cto "$gpu" "$util" "${prefix}_step4/results_dedup" "${prefix}_step5/results"
  fi
  echo "  iter2 $(_count "${prefix}_step5/results")/${END_INDEX}"
  _score_agent 2 "${prefix}_step5/results"

  _distill_round "$gpu" "$util" 6 "${prefix}_step5/results" "${prefix}_step4/results_dedup" "$prefix"
  if [ "$method" = "rse" ]; then
    _guided_rse "$gpu" "$util" "${prefix}_step6/results_dedup" "${prefix}_step7/results"
  else
    _guided_cto "$gpu" "$util" "${prefix}_step6/results_dedup" "${prefix}_step7/results"
  fi
  echo "  iter3 $(_count "${prefix}_step7/results")/${END_INDEX}"
  _score_agent 3 "${prefix}_step7/results"

  python - "$method" "$prefix" <<'PY'
import json, os, sys
method, prefix = sys.argv[1], sys.argv[2]
link = prefix + "_step1/results"
summary = {
    "dataset": "TravelPlanner_Val60",
    "model": "Phi-4-Reasoning",
    "method": method.upper(),
    "shared_iter0": os.path.realpath(link),
    "steps": {
        "iter0": link,
        "iter1": prefix + "_step3/results",
        "iter2": prefix + "_step5/results",
        "iter3": prefix + "_step7/results",
    },
}
out = os.path.join(os.path.dirname(prefix), "run0_eval_summary.json")
json.dump(summary, open(out, "w"), indent=2)
print(f"Wrote {out}")
PY
  echo "========== ${method} done at $(date '+%F %T') =========="
}

if [ "${1:-}" = "--resume-cto" ]; then
  echo "========== Resume CTO from existing step2 | GPU ${GPU_CTO} =========="
  echo "Started at $(date '+%F %T')"
  _run_method cto "$GPU_CTO" "$CTO_PREFIX"
  echo "Resume CTO finished at $(date '+%F %T')"
  exit 0
fi

echo "========== TravelPlanner Phi-4 | shared iter0 on GPU ${GPU_RSE}+${GPU_CTO}, then RSE/CTO =========="
echo "QUESTION_FILE=${QUESTION_FILE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SHARED_PREFIX=${SHARED_PREFIX}"
echo "RSE_PREFIX=${RSE_PREFIX}  CTO_PREFIX=${CTO_PREFIX}"
echo "Started at $(date '+%F %T')"

if [ "$(_count "${SHARED_PREFIX}_step1/results")" -ge "$END_INDEX" ]; then
  echo "[skip] shared iter0 already has $(_count "${SHARED_PREFIX}_step1/results")/${END_INDEX}"
else
  mkdir -p "/data/ppnm/tmp/tp_phi4_iter0_gpu${GPU_RSE}_$$" "/data/ppnm/tmp/tp_phi4_iter0_gpu${GPU_CTO}_$$"
  _sample_shard "$GPU_RSE" 0 "$MID_INDEX" >"${LOG_DIR}/iter0_gpu${GPU_RSE}.log" 2>&1 &
  pid_a=$!
  _sample_shard "$GPU_CTO" "$MID_INDEX" "$END_INDEX" >"${LOG_DIR}/iter0_gpu${GPU_CTO}.log" 2>&1 &
  pid_b=$!
  echo "iter0 PIDs GPU${GPU_RSE}=${pid_a} GPU${GPU_CTO}=${pid_b}"
  set +e
  wait "$pid_a"
  st_a=$?
  wait "$pid_b"
  st_b=$?
  set -e
  echo "iter0 exit GPU${GPU_RSE}=${st_a} GPU${GPU_CTO}=${st_b} files=$(_count "${SHARED_PREFIX}_step1/results")/${END_INDEX}"
  if [ "$st_a" -ne 0 ] || [ "$st_b" -ne 0 ]; then
    echo "ERROR: shared iter0 failed. See ${LOG_DIR}/iter0_gpu*.log"
    exit 1
  fi
  if [ "$(_count "${SHARED_PREFIX}_step1/results")" -lt "$END_INDEX" ]; then
    echo "ERROR: shared iter0 produced $(_count "${SHARED_PREFIX}_step1/results") files, expected ${END_INDEX}"
    exit 1
  fi
fi

_score_agent 0 "${SHARED_PREFIX}_step1/results"

_run_method rse "$GPU_RSE" "$RSE_PREFIX" >"${LOG_DIR}/rse_gpu${GPU_RSE}.log" 2>&1 &
pid_rse=$!
_run_method cto "$GPU_CTO" "$CTO_PREFIX" >"${LOG_DIR}/cto_gpu${GPU_CTO}.log" 2>&1 &
pid_cto=$!
echo "iter1-3 PIDs RSE=${pid_rse} CTO=${pid_cto}"
echo "RSE log: ${LOG_DIR}/rse_gpu${GPU_RSE}.log"
echo "CTO log: ${LOG_DIR}/cto_gpu${GPU_CTO}.log"
set +e
wait "$pid_rse"
st_rse=$?
wait "$pid_cto"
st_cto=$?
set -e
echo "Finished at $(date '+%F %T')  RSE=${st_rse} CTO=${st_cto}"
if [ "$st_rse" -ne 0 ] || [ "$st_cto" -ne 0 ]; then
  exit 1
fi
