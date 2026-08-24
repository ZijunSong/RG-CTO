#!/usr/bin/env python3
"""Aggregate RG-CTO 3-run summaries and patch RG-CTO Method Core.md table."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "—"
    m = statistics.mean(values)
    if len(values) == 1:
        return f"{m:.1f}"
    s = statistics.stdev(values)
    return f"{m:.1f}±{s:.1f}"


def load_run_summaries(runs_root: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"iter0": [], "iter1": [], "iter2": []}
    for run_id in range(3):
        p = runs_root / f"run{run_id}_eval_summary.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        for key in out:
            if data.get(key) is not None:
                out[key].append(float(data[key]))
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    datasets = {
        "HMMT24": project_root / "results/runs/HMMT24_Qwen3_30B_A3B_Thinking_2507_RG_CTO",
        "HMMT25": project_root / "results/runs/HMMT25_Qwen3_30B_A3B_Thinking_2507_RG_CTO",
        "HLE": project_root / "results/runs/HLE_math_text_Qwen3_30B_A3B_Thinking_2507_RG_CTO",
    }

    agg: dict[str, dict[str, str]] = {}
    for name, root in datasets.items():
        vals = load_run_summaries(root)
        agg[name] = {k: fmt_mean_std(v) for k, v in vals.items()}

    md_file = project_root / "RG-CTO Method Core.md"
    text = md_file.read_text(encoding="utf-8")
    old = (
        "| Qwen3-30B-A3B-Thinking-2507 | RG-CTO | 55.8 | — | — | 66.2 | — | — | "
        "23.4 | — | — |"
    )
    new = (
        f"| Qwen3-30B-A3B-Thinking-2507 | RG-CTO | "
        f"{agg['HMMT24']['iter0']} | {agg['HMMT24']['iter1']} | {agg['HMMT24']['iter2']} | "
        f"{agg['HMMT25']['iter0']} | {agg['HMMT25']['iter1']} | {agg['HMMT25']['iter2']} | "
        f"{agg['HLE']['iter0']} | {agg['HLE']['iter1']} | {agg['HLE']['iter2']} |"
    )
    if old not in text:
        print("ERROR: Could not find RG-CTO 30B table row to patch", file=sys.stderr)
        sys.exit(1)
    text = text.replace(old, new, 1)

    note_marker = "RG-CTO 待跑"
    if note_marker in text:
        text = text.replace(
            note_marker,
            "RG-CTO 30B 三数据集 3-run 已完成（`results/runs/*_RG_CTO/`）",
            1,
        )

    md_file.write_text(text, encoding="utf-8")
    summary_path = project_root / "results/summaries/rg_cto_30b_mean_std.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched {md_file}")
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
