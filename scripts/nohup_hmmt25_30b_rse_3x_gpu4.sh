#!/bin/bash
# HMMT25 × Qwen3-30B-Thinking × RSE: 3 runs on GPU4, shared lowest iter0, iter1-2 only.
# After completion: aggregate mean±std and patch RG-CTO Method Core.md table.
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

export CUDA_VISIBLE_DEVICES=4
export TENSOR_PARALLEL_SIZE=1

RUNS_ROOT="${PROJECT_ROOT}/results/runs/HMMT25_Qwen3_30B_A3B_Thinking_2507_RSE"
STEP1_RESULTS_SRC="${STEP1_RESULTS_SRC:-/data/ppnm/Contrastive-Trajectory-Optimization/runs/HMMT25_30B_CTO_vllm_step1/results}"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$RUNS_ROOT" "$LOG_DIR"

echo "========== HMMT25 30B RSE 3× | GPU4 | shared iter0 =========="
echo "RUNS_ROOT=${RUNS_ROOT}"
echo "STEP1_RESULTS_SRC=${STEP1_RESULTS_SRC}"
echo "Started at $(date '+%F %T')"

for run_id in 0 1 2; do
  OUT_PREFIX="${RUNS_ROOT}/run${run_id}"
  if [ -f "${OUT_PREFIX}_eval_summary.json" ]; then
    echo "[skip] run${run_id} already has eval summary"
    continue
  fi
  echo ""
  echo ">>>>>>>>>> Run ${run_id}/2 <<<<<<<<<<"
  export OUT_PREFIX STEP1_RESULTS_SRC
  bash "${SCRIPT_DIR}/run_rse_hmmt25_30b_from_iter0_iter2.sh" 0 30
done

SUMMARY_JSON="${RUNS_ROOT}/mean_std_summary.json"
python - "$RUNS_ROOT" "$SUMMARY_JSON" <<'PY'
import json, statistics, sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out_json = Path(sys.argv[2])
rows = []
for run_id in range(3):
    p = runs_root / f"run{run_id}_eval_summary.json"
    if not p.exists():
        raise SystemExit(f"Missing {p}")
    d = json.load(open(p))
    rows.append(d)

summary = {"runs": rows, "mean_std": {}}
for it in ("iter0", "iter1", "iter2"):
    vals = [r[it] for r in rows if r.get(it) is not None]
    if not vals:
        continue
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    summary["mean_std"][it] = {
        "mean": round(m, 1),
        "std": round(s, 1),
        "values": vals,
        "n": len(vals),
    }

json.dump(summary, open(out_json, "w"), indent=2)
print(json.dumps(summary["mean_std"], indent=2))
PY

MD_FILE="${PROJECT_ROOT}/RG-CTO Method Core.md"
python - "$RUNS_ROOT" "$SUMMARY_JSON" "$MD_FILE" <<'PY'
import json, re, sys
from pathlib import Path

runs_root, summary_json, md_file = map(Path, sys.argv[1:4])
summary = json.load(open(summary_json))
ms = summary["mean_std"]

def fmt(it):
    x = ms[it]
    if x["n"] <= 1:
        return f"{x['mean']:.1f}"
    return f"{x['mean']:.1f}±{x['std']:.1f}"

iter0 = fmt("iter0")
iter1 = fmt("iter1")
iter2 = fmt("iter2")

text = md_file.read_text(encoding="utf-8")
old = "| Qwen3-30B-A3B-Thinking-2507 | RSE | 55.8 | 68.8±2.7 | 70.6±3.4 | 66.2 | 66.7 | **80.0** | 23.4 | 38.8±1.4 | 40.8±2.0 |"
new = f"| Qwen3-30B-A3B-Thinking-2507 | RSE | 55.8 | 68.8±2.7 | 70.6±3.4 | {iter0} | {iter1} | {iter2} | 23.4 | 38.8±1.4 | 40.8±2.0 |"
if old not in text:
    raise SystemExit("Could not find 30B RSE table row to patch")
text = text.replace(old, new, 1)

note_old = "> 注：iter0 为所有方法共享的 baseline rollout。HMMT24 / HLE 的 iter1/iter2 为 3-seed 均值 ± 标准差（RSE 来自 `RSE/runs/reproduce/`，CTO 来自 3-run vllm）。HMMT25 Thinking 的 CTO 为 CTO-base（`nocross_llm_judge`）单 run；RSE 为同协议对照实验（30B/4B 来自 CTO 论文主表，Instruct 来自附录）。Phi-4 HMMT25 仅有 RSE iter0（21.7%）。RG-CTO 待跑。"
note_new = (
    f"> 注：iter0 为所有方法共享的 baseline rollout。HMMT24 / HLE 的 iter1/iter2 为 3-seed 均值 ± 标准差（RSE 来自 `RSE/runs/reproduce/`，CTO 来自 3-run vllm）。"
    f"HMMT25 30B RSE：3-run（`results/runs/HMMT25_Qwen3_30B_A3B_Thinking_2507_RSE/`，共享最低 iter0 `HMMT25_30B_CTO_vllm_step1`≈64.9%）。"
    f"HMMT25 30B CTO 为 CTO-base + vllm 合并 3-run（见上文）。其他 HMMT25 行来源见 iter0 表。RG-CTO 待跑。"
)
if note_old in text:
    text = text.replace(note_old, note_new, 1)

md_file.write_text(text, encoding="utf-8")
print(f"Patched {md_file}")
print(f"HMMT25 RSE 30B: iter0={iter0}, iter1={iter1}, iter2={iter2}")
PY

echo ""
echo "========== All done at $(date '+%F %T') =========="
echo "Summary: ${SUMMARY_JSON}"
echo "Runs: ${RUNS_ROOT}"
