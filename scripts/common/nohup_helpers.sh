#!/bin/bash
# Shared helpers for nohup experiment wrappers.

run_has_valid_summary() {
  local prefix="$1"
  python - "${prefix}_eval_summary.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    sys.exit(0 if all(d.get(k) is not None for k in ("iter0", "iter1", "iter2")) else 1)
except Exception:
    sys.exit(1)
PY
}
