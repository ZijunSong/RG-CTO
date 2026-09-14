#!/bin/bash
# RG-CTO (Qwen3-4B-Thinking): reuse frozen iter0, run iter1-2 (step2→step5).
# Required env: OUT_PREFIX, STEP1_RESULTS_SRC, QUESTION_FILE
# Optional: CUDA_VISIBLE_DEVICES, DATASET (default HMMT24), END_INDEX
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-HMMT24}"
export MODEL_NAME="${MODEL_NAME:-/data/ppnm/models/Qwen3-4B-Thinking-2507}"
export MAX_TOKENS="${MAX_TOKENS:-38912}"
export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-8192}"
export N_EXP_COMPLETIONS="${N_EXP_COMPLETIONS:-48}"
export DISTILL_MAX_MODEL_LEN="${DISTILL_MAX_MODEL_LEN:-100000}"
export RGCTO_MAX_MODEL_LEN="${RGCTO_MAX_MODEL_LEN:-100000}"
export RGCTO_GPU_MEMORY_UTILIZATION="${RGCTO_GPU_MEMORY_UTILIZATION:-0.60}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"
export CTO_RETRIEVAL_RERANK_POOL_MULT="${CTO_RETRIEVAL_RERANK_POOL_MULT:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-/data/ppnm/tmp/rg_cto_qwen3_4b_gpu${CUDA_VISIBLE_DEVICES:-0}_$$}"

exec bash "${SCRIPT_DIR}/run_rg_cto_phi4_from_iter0_iter2.sh" "$@"
