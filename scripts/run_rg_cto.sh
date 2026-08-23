#!/bin/bash
# RG-CTO pipeline: same as CTO but steps 3/5/7 use rg_cto_guided_search.py
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/run_rg_cto.sh <start_index> <end_index>"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

pip install sentence-transformers -q

BATCH_SIZE="${BATCH_SIZE:-30}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
N_COMPLETIONS="${N_COMPLETIONS:-32}"
MAX_TOKENS="${MAX_TOKENS:-38912}"
N_EXPERIENCE_COMPLETIONS="${N_EXPERIENCE_COMPLETIONS:-32}"
THRESHOLD="${THRESHOLD:-0.8}"

ALPHA="${ALPHA:-0.7}"
GATE_DELTA="${GATE_DELTA:-0.15}"
PILOT_N="${PILOT_N:-4}"

: "${MODEL_NAME:?Set MODEL_NAME}"
: "${QUESTION_FILE:?Set QUESTION_FILE}"
: "${OUT_PREFIX:?Set OUT_PREFIX}"
EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"

STEP_1="${OUT_PREFIX}_step1"
STEP_2="${OUT_PREFIX}_step2"
STEP_3="${OUT_PREFIX}_step3"
STEP_4="${OUT_PREFIX}_step4"
STEP_5="${OUT_PREFIX}_step5"
STEP_6="${OUT_PREFIX}_step6"
STEP_7="${OUT_PREFIX}_step7"

mkdir -p "${STEP_1}/results" "${STEP_3}/results" "${STEP_5}/results" "${STEP_7}/results"
mkdir -p "${STEP_2}/results" "${STEP_2}/results_dedup" "${STEP_2}/results_dedup_debug"
mkdir -p "${STEP_4}/results" "${STEP_4}/results_dedup" "${STEP_4}/results_dedup_debug"
mkdir -p "${STEP_6}/results" "${STEP_6}/results_dedup" "${STEP_6}/results_dedup_debug"

python code/standard_sampling.py \
    --model "$MODEL_NAME" \
    --input "$QUESTION_FILE" \
    --output "${STEP_1}/results" \
    --n-completions "$N_COMPLETIONS" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --start-idx "$START_INDEX" \
    --end-idx "$END_INDEX"

for round in 2 4 6; do
    prev_answer="${OUT_PREFIX}_step$((round - 1))/results"
    exp_dir="${OUT_PREFIX}_step${round}"
    prev_dedup=""
    if [ "$round" -gt 2 ]; then
        prev_dedup="${OUT_PREFIX}_step$((round - 2))/results_dedup"
    fi
    python code/experience_distillation.py \
        --model "$MODEL_NAME" \
        --question-file "$QUESTION_FILE" \
        --answer-dir "$prev_answer" \
        --output-dir "${exp_dir}/results" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --top-k "$TOP_K" \
        --max-tokens "$MAX_TOKENS" \
        --n-samples 1 \
        --start-idx "$START_INDEX" \
        --end-idx "$END_INDEX"
    dedup_args=(--experience-dir "${exp_dir}/results" --output-dir "${exp_dir}/results_dedup"
        --debug-dir "${exp_dir}/results_dedup_debug" --model-path "$EMB_MODEL"
        --threshold "$THRESHOLD" --keep-order)
    if [ -n "$prev_dedup" ]; then
        dedup_args+=(--previous-experience-dir "$prev_dedup")
    fi
    python code/experience_dedup.py "${dedup_args[@]}"
done

for guided_step in 3 5 7; do
    exp_step=$((guided_step - 1))
    python code/rg_cto_guided_search.py \
        --model "$MODEL_NAME" \
        --input "$QUESTION_FILE" \
        --experience-dir "${OUT_PREFIX}_step${exp_step}/results_dedup" \
        --output "${OUT_PREFIX}_step${guided_step}/results" \
        --n-experience-completions "$N_EXPERIENCE_COMPLETIONS" \
        --n-completions "$N_COMPLETIONS" \
        --alpha "$ALPHA" \
        --gate-delta "$GATE_DELTA" \
        --pilot-n "$PILOT_N" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --top-k "$TOP_K" \
        --max-tokens "$MAX_TOKENS" \
        --start-idx "$START_INDEX" \
        --end-idx "$END_INDEX"
done

echo "RG-CTO pipeline finished. Final results: ${STEP_7}/results"
