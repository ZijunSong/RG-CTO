#!/bin/bash
# RSE baseline pipeline (steps 3/5/7 use experience_guided_search.py)
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/run_rse.sh <start_index> <end_index>"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_NAME:?Set MODEL_NAME}"
: "${QUESTION_FILE:?Set QUESTION_FILE}"
: "${OUT_PREFIX:?Set OUT_PREFIX}"

export START_INDEX END_INDEX
bash scripts/common/rse_full_pipeline.sh "$START_INDEX" "$END_INDEX"
