#!/bin/bash
# Monitor CTO/RG-CTO iter1 (step3) completion and compute pass@1.
set -euo pipefail

OUT="/data/ppnm/RG-CTO/logs/monitor_iter1_status.txt"
RESULT="/data/ppnm/RG-CTO/logs/monitor_iter1_results.json"

CTO_DIR="/data/ppnm/RG-CTO/results/runs/CodeContests_Phi_4_Reasoning_CTO/run0_step3/results"
RG_DIR="/data/ppnm/RG-CTO/results/runs/CodeContests_Phi_4_Reasoning_RG_CTO/run0_step3/results"

is_complete() {
  local d="$1"
  [ -f "$d/6.json" ] && [ -f "$d/22.json" ] && \
  [ "$(find "$d" -maxdepth 1 -name '*.json' | wc -l)" -ge 165 ]
}

while true; do
  ts="$(date '+%F %T')"
  cto_n=$(find "$CTO_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  rg_n=$(find "$RG_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  cto_done=false; rg_done=false
  is_complete "$CTO_DIR" && cto_done=true
  is_complete "$RG_DIR" && rg_done=true
  echo "[$ts] CTO=${cto_n}/165 done=$cto_done | RG-CTO=${rg_n}/165 done=$rg_done" | tee -a "$OUT"

  if $cto_done && $rg_done; then
    echo "[$ts] BOTH COMPLETE - running eval" | tee -a "$OUT"
    source /data/ppnm/miniconda3/etc/profile.d/conda.sh
    conda activate cto
    cd /tmp
    python3 /data/ppnm/RG-CTO/scripts/eval_codecontests_pass1.py \
      "$CTO_DIR" "CTO iter1" \
      "$RG_DIR" "RG-CTO iter1" \
      > "$RESULT" 2>&1
    echo "[$ts] EVAL DONE" | tee -a "$OUT"
    cat "$RESULT" | tee -a "$OUT"
    echo "=== ITER1_MONITOR_COMPLETE ===" | tee -a "$OUT"
    exit 0
  fi
  sleep 120
done
