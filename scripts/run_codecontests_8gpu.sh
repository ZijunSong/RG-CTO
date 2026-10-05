#!/bin/bash
# CodeContests Test 165 on eight GPUs.
# Same iterative protocol as the math benchmarks: iter0 sampling, then
# distill / dedup / guided search through iter3. One vLLM engine per GPU,
# disjoint question shards, shared output directory. After each generation
# step the saved programs are executed against the public and private tests.
#
# Usage:
#   bash scripts/run_codecontests_8gpu.sh
#   METHOD=cto bash scripts/run_codecontests_8gpu.sh
#   METHOD=rg_cto MODEL_NAME=/data/ppnm/models/Phi-4-reasoning \
#     bash scripts/run_codecontests_8gpu.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 bash scripts/run_codecontests_8gpu.sh
#
# METHOD is rse, cto, or rg_cto. NGPU defaults to 8.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate cto

export METHOD="${METHOD:-rse}"
export NGPU="${NGPU:-8}"
export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Qwen3-4B-Thinking-2507}"
export QUESTION_FILE="${QUESTION_FILE:-${PROJECT_ROOT}/data/CodeContests_Test_165.jsonl}"
export DATASET="${DATASET:-CodeContests}"
export EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"
export RETRIEVAL_RERANK_MODEL="${RETRIEVAL_RERANK_MODEL:-/data/ppnm/models/cross-encoder-ms-marco-MiniLM-L-6-v2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_TAG="$(basename "$MODEL_NAME")"
export OUT_PREFIX="${OUT_PREFIX:-${PROJECT_ROOT}/results/runs/CodeContests_${MODEL_TAG}_${METHOD}/run0}"

echo "========== CodeContests 8-GPU ${METHOD} =========="
echo "MODEL_NAME=${MODEL_NAME}"
echo "QUESTION_FILE=${QUESTION_FILE}"
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "NGPU=${NGPU}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0-7}"
echo "Started at $(date '+%F %T')"

bash "${SCRIPT_DIR}/run_8gpu.sh" 0 165

echo "========== CodeContests finished at $(date '+%F %T') =========="
