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

# Defaults match the local Qwen3 RG-CTO runs (run_rg_cto_qwen3_4b_from_iter0_iter2.sh).
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
N_COMPLETIONS="${N_COMPLETIONS:-32}"
MAX_TOKENS="${MAX_TOKENS:-38912}"
DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-8192}"
N_EXPERIENCE_COMPLETIONS="${N_EXPERIENCE_COMPLETIONS:-48}"
THRESHOLD="${THRESHOLD:-0.85}"
EXPERIENCE_JUDGE_MODE="${EXPERIENCE_JUDGE_MODE:-llm_judge}"

ALPHA="${ALPHA:-0.7}"
GATE_DELTA="${GATE_DELTA:-0.4}"
TAU_MATCH="${TAU_MATCH:-0.8}"
LAMBDA_U="${LAMBDA_U:-0.5}"
LAMBDA_L="${LAMBDA_L:-0.5}"
PILOT_N="${PILOT_N:-4}"
MIN_PITFALL_SUPPORT="${MIN_PITFALL_SUPPORT:-2}"
CTO_AGG_MAX_PROP="${CTO_AGG_MAX_PROP:-96}"
CTO_AGG_MAX_PIT="${CTO_AGG_MAX_PIT:-96}"
CTO_RETRIEVAL_RERANK_POOL_MULT="${CTO_RETRIEVAL_RERANK_POOL_MULT:-8}"
DISTILL_MAX_MODEL_LEN="${DISTILL_MAX_MODEL_LEN:-100000}"
DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"
DISTILL_MAX_NUM_SEQS="${DISTILL_MAX_NUM_SEQS:-128}"
RGCTO_MAX_MODEL_LEN="${RGCTO_MAX_MODEL_LEN:-100000}"
RGCTO_GPU_MEMORY_UTILIZATION="${RGCTO_GPU_MEMORY_UTILIZATION:-0.60}"
MAX_PILOT_TOKENS="${MAX_PILOT_TOKENS:-$MAX_TOKENS}"

: "${MODEL_NAME:?Set MODEL_NAME}"
: "${QUESTION_FILE:?Set QUESTION_FILE}"
: "${OUT_PREFIX:?Set OUT_PREFIX}"
EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"
RETRIEVAL_RERANK_MODEL="${RETRIEVAL_RERANK_MODEL:-/data/ppnm/models/cross-encoder-ms-marco-MiniLM-L-6-v2}"

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
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$RGCTO_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$RGCTO_GPU_MEMORY_UTILIZATION" \
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
    prev_answer="${OUT_PREFIX}_step$((guided_step - 2))/results"
    python code/rg_cto_guided_search.py \
        --model "$MODEL_NAME" \
        --input "$QUESTION_FILE" \
        --experience-dir "${OUT_PREFIX}_step${exp_step}/results_dedup" \
        --answer-dir "$prev_answer" \
        --output "${OUT_PREFIX}_step${guided_step}/results" \
        --n-experience-completions "$N_EXPERIENCE_COMPLETIONS" \
        --n-completions "$N_COMPLETIONS" \
        --alpha "$ALPHA" \
        --gate-delta "$GATE_DELTA" \
        --tau-match "$TAU_MATCH" \
        --lambda-u "$LAMBDA_U" \
        --lambda-l "$LAMBDA_L" \
        --pilot-n "$PILOT_N" \
        --min-pitfall-support "$MIN_PITFALL_SUPPORT" \
        --max-pilot-tokens "$MAX_PILOT_TOKENS" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$RGCTO_MAX_MODEL_LEN" \
        --gpu-memory-utilization "$RGCTO_GPU_MEMORY_UTILIZATION" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --top-k "$TOP_K" \
        --max-tokens "$MAX_TOKENS" \
        --experience-retrieval embedding_rerank \
        --retrieval-embedding-model "$EMB_MODEL" \
        --retrieval-rerank-model "$RETRIEVAL_RERANK_MODEL" \
        --retrieval-rerank-pool-mult "$CTO_RETRIEVAL_RERANK_POOL_MULT" \
        --max-aggregated-propositions "$CTO_AGG_MAX_PROP" \
        --max-aggregated-pitfalls "$CTO_AGG_MAX_PIT" \
        --start-idx "$START_INDEX" \
        --end-idx "$END_INDEX"
done

echo "RG-CTO pipeline finished. Final results: ${STEP_7}/results"
