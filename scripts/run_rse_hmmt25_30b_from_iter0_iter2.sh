#!/bin/bash
# RSE on HMMT25 (Qwen3-30B-Thinking): reuse frozen iter0, run iter1-2 (step2→step5).
# Required env: OUT_PREFIX, STEP1_RESULTS_SRC
# Optional: CUDA_VISIBLE_DEVICES (default 4), TENSOR_PARALLEL_SIZE (default 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export CONDA_ROOT="${CONDA_ROOT:-/data/ppnm/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-cto}"
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi
export PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):${PATH}"
command -v python >/dev/null 2>&1 || { echo "ERROR: python not found (expected $PYTHON_BIN)"; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

START_INDEX="${1:-0}"
END_INDEX="${2:-30}"

: "${OUT_PREFIX:?OUT_PREFIX is required}"
: "${STEP1_RESULTS_SRC:?STEP1_RESULTS_SRC is required}"

export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Qwen3-30B-A3B-Thinking-2507}"
export QUESTION_FILE="${QUESTION_FILE:-${PROJECT_ROOT}/data/HMMT_25.jsonl}"

export BATCH_SIZE="${BATCH_SIZE:-2048}"
export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-8192}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export N_COMPLETIONS="${N_COMPLETIONS:-32}"
export MAX_TOKENS="${MAX_TOKENS:-38912}"
export N_EXP_COMPLETIONS="${N_EXP_COMPLETIONS:-32}"
export THRESHOLD="${THRESHOLD:-0.8}"
export EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"
export EXPERIENCE_JUDGE_MODE="${EXPERIENCE_JUDGE_MODE:-llm_judge}"

export DISTILL_MAX_MODEL_LEN="${DISTILL_MAX_MODEL_LEN:-65536}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.95}"
export DISTILL_MAX_NUM_SEQS="${DISTILL_MAX_NUM_SEQS:-256}"
export EGS_MAX_MODEL_LEN="${EGS_MAX_MODEL_LEN:-$DISTILL_MAX_MODEL_LEN}"
export EGS_GPU_MEMORY_UTILIZATION="${EGS_GPU_MEMORY_UTILIZATION:-$DISTILL_GPU_MEMORY_UTILIZATION}"

export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export TMPDIR="${TMPDIR:-/data/tmp/rg_cto_rse_hmmt25_$$}"
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/ppnm/.cache}"
mkdir -p "$XDG_CACHE_HOME"

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
  python - "$ver_dir" <<'PY'
import json, sys
ver_dir = sys.argv[1]
with open(ver_dir + "/pass_at_1.json", "r", encoding="utf-8") as f:
    m = json.load(f)
v = m.get("pass_at_k", {}).get("pass@1", None)
print("N/A" if v is None else f"{v*100:.2f}%")
PY
}

echo "========== RSE HMMT25 30B iter0→iter2 | GPU=${CUDA_VISIBLE_DEVICES} TP=${TENSOR_PARALLEL_SIZE} =========="
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "STEP1_RESULTS_SRC=${STEP1_RESULTS_SRC}"
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
rse_pass1_iter 0 "${OUT_PREFIX}_step1/results" | sed 's/^/  /'

echo "---------- Step 2: Experience Distillation ----------"
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
  --end-idx "$END_INDEX"

echo "---------- Step 2.5: Experience Deduplication ----------"
mkdir -p "${OUT_PREFIX}_step2/results_dedup" "${OUT_PREFIX}_step2/results_dedup_debug"
python code/experience_dedup.py \
  --experience-dir "${OUT_PREFIX}_step2/results" \
  --output-dir "${OUT_PREFIX}_step2/results_dedup" \
  --debug-dir "${OUT_PREFIX}_step2/results_dedup_debug" \
  --model-path "$EMB_MODEL" \
  --threshold "$THRESHOLD" \
  --keep-order

echo "---------- Step 3: Experience-Guided Search (iter1) ----------"
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
  --end-idx "$END_INDEX"

echo "---------- Step 4: Experience Distillation ----------"
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

echo "---------- Step 5: Experience-Guided Search (iter2) ----------"
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
  --end-idx "$END_INDEX"

echo "---------- Pass@1 summary (iter0-2) ----------"
for iter in 0 1 2; do
  case "$iter" in
    0) dir="${OUT_PREFIX}_step1/results" ;;
    1) dir="${OUT_PREFIX}_step3/results" ;;
    2) dir="${OUT_PREFIX}_step5/results" ;;
  esac
  echo -n "  iter${iter}  "
  rse_pass1_iter "$iter" "$dir"
done

python - "$OUT_PREFIX" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
summary = {}
for it, step in [(0,1),(1,3),(2,5)]:
    f = out / f"step{step}" / "results" / "pass_at_1.json"
    if f.exists():
        d = json.load(open(f))
        v = d.get("pass_at_k", {}).get("pass@1")
        summary[f"iter{it}"] = round(v*100, 2) if v is not None else None
out_json = out.parent / f"{out.name}_eval_summary.json"
json.dump(summary, open(out_json, "w"), indent=2)
print(f"Wrote {out_json}: {summary}")
PY

echo "========== Done: ${OUT_PREFIX} =========="
