#!/bin/bash
# Rerun missing CodeContests indices: step2 distill -> step2.5 dedup -> step3 guided search.
# Usage: rerun_missing_step3_phi4.sh <cto|rg_cto> <gpu_id> <idx1> [idx2 ...]
set -euo pipefail

METHOD="${1:?method: cto or rg_cto}"
GPU="${2:?gpu id}"
shift 2
INDICES=("$@")
if [ "${#INDICES[@]}" -eq 0 ]; then
  echo "Usage: $0 <cto|rg_cto> <gpu_id> <idx1> [idx2 ...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export CUDA_VISIBLE_DEVICES="$GPU"
export TENSOR_PARALLEL_SIZE=1
export DATASET=CodeContests
export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Phi-4-reasoning}"
export QUESTION_FILE="${PROJECT_ROOT}/data/CodeContests_Test_165.jsonl"
export EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"
export RETRIEVAL_RERANK_MODEL="${RETRIEVAL_RERANK_MODEL:-/data/ppnm/models/cross-encoder-ms-marco-MiniLM-L-6-v2}"
export THRESHOLD="${THRESHOLD:-0.85}"
export EXPERIENCE_JUDGE_MODE="${EXPERIENCE_JUDGE_MODE:-llm_judge}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-/data/ppnm/tmp/rerun_${METHOD}_gpu${GPU}_$$}"
mkdir -p /data/ppnm/tmp "$TMPDIR"

case "$METHOD" in
  cto)
    OUT_PREFIX="${PROJECT_ROOT}/results/runs/CodeContests_Phi_4_Reasoning_CTO/run0"
    SEARCH_SCRIPT="code/cto_guided_search.py"
    GPU_UTIL="${CTO_GPU_MEMORY_UTILIZATION:-0.70}"
    SEARCH_EXTRA=(
      --alpha 0.7
      --plausibility-top-k 20
      --experience-retrieval embedding_rerank
      --retrieval-embedding-model "$EMB_MODEL"
      --retrieval-rerank-model "$RETRIEVAL_RERANK_MODEL"
      --retrieval-rerank-pool-mult 4
      --max-aggregated-propositions 96
      --max-aggregated-pitfalls 96
    )
    ;;
  rg_cto)
    OUT_PREFIX="${PROJECT_ROOT}/results/runs/CodeContests_Phi_4_Reasoning_RG_CTO/run0"
    SEARCH_SCRIPT="code/rg_cto_guided_search.py"
    GPU_UTIL="${RGCTO_GPU_MEMORY_UTILIZATION:-0.70}"
    SEARCH_EXTRA=(
      --alpha 0.7
      --gate-delta 0.4
      --tau-match 0.8
      --lambda-u 0.5
      --lambda-l 0.5
      --pilot-n 4
      --answer-dir "${OUT_PREFIX}_step1/results"
      --min-pitfall-support 2
      --experience-retrieval embedding_rerank
      --retrieval-embedding-model "$EMB_MODEL"
      --retrieval-rerank-model "$RETRIEVAL_RERANK_MODEL"
      --retrieval-rerank-pool-mult 4
      --max-aggregated-propositions 96
      --max-aggregated-pitfalls 96
    )
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    exit 1
    ;;
esac

STEP1_DIR="${OUT_PREFIX}_step1/results"
STEP2_DIR="${OUT_PREFIX}_step2/results"
STEP2_DEDUP_DIR="${OUT_PREFIX}_step2/results_dedup"
STEP3_DIR="${OUT_PREFIX}_step3/results"

echo "========== Rerun ${METHOD} step2→step3 | GPU=${GPU} | indices: ${INDICES[*]} =========="
echo "Started at $(date '+%F %T')"

for idx in "${INDICES[@]}"; do
  next=$((idx + 1))
  echo "---------- idx ${idx}: step2 distill ----------"
  mkdir -p "$STEP2_DIR"
  python code/experience_distillation.py \
    --model "$MODEL_NAME" \
    --question-file "$QUESTION_FILE" \
    --answer-dir "$STEP1_DIR" \
    --output-dir "$STEP2_DIR" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization "$DISTILL_GPU_MEMORY_UTILIZATION" \
    --max-num-seqs 128 \
    --batch-size 2048 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --max-tokens 32768 \
    --n-samples 1 \
    --experience_judge_mode "$EXPERIENCE_JUDGE_MODE" \
    --dataset "$DATASET" \
    --start-idx "$idx" \
    --end-idx "$next"

  echo "---------- idx ${idx}: step2.5 dedup ----------"
  mkdir -p "$STEP2_DEDUP_DIR" "${OUT_PREFIX}_step2/results_dedup_debug"
  python code/experience_dedup.py \
    --experience-dir "$STEP2_DIR" \
    --output-dir "$STEP2_DEDUP_DIR" \
    --debug-dir "${OUT_PREFIX}_step2/results_dedup_debug" \
    --model-path "$EMB_MODEL" \
    --threshold "$THRESHOLD" \
    --keep-order

  echo "---------- idx ${idx}: step3 guided search ----------"
  mkdir -p "$STEP3_DIR"
  python "$SEARCH_SCRIPT" \
    --model "$MODEL_NAME" \
    --input "$QUESTION_FILE" \
    --experience-dir "$STEP2_DEDUP_DIR" \
    --output "$STEP3_DIR" \
    --n-experience-completions 32 \
    --n-completions 32 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --max-tokens 32768 \
    --dataset "$DATASET" \
    --start-idx "$idx" \
    --end-idx "$next" \
    "${SEARCH_EXTRA[@]}"
done

echo "========== Done ${METHOD} rerun at $(date '+%F %T') =========="
