#!/usr/bin/env python3
"""Aggregate Phi-4 RG-CTO 3-run summaries and patch RG-CTO Method Core.md tables."""
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
        "BambooQA": project_root / "results/runs/BambooQA_Phi_4_Reasoning_RG_CTO",
        "HMMT24": project_root / "results/runs/HMMT24_Phi_4_Reasoning_RG_CTO",
        "HLE": project_root / "results/runs/HLE_math_text_Phi_4_Reasoning_RG_CTO",
    }

    agg: dict[str, dict[str, str]] = {}
    for name, root in datasets.items():
        if not root.exists():
            print(f"Skip {name}: {root} not found", file=sys.stderr)
            continue
        vals = load_run_summaries(root)
        agg[name] = {k: fmt_mean_std(v) for k, v in vals.items()}

    if not agg:
        print("No completed Phi-4 RG-CTO runs to patch", file=sys.stderr)
        sys.exit(0)

    md_file = project_root / "RG-CTO Method Core.md"
    text = md_file.read_text(encoding="utf-8")

    if "BambooQA" in agg:
        old_bamboo = (
            "| RG-CTO | 35.7 | — | — | — | 35.7 | — | — | — | 35.7 | — | — | — | 35.7 | — | — | — |"
        )
        new_bamboo = (
            f"| RG-CTO | {agg['BambooQA']['iter0']} | {agg['BambooQA']['iter1']} | "
            f"{agg['BambooQA']['iter2']} | — | "
            f"{agg['BambooQA']['iter0']} | {agg['BambooQA']['iter1']} | {agg['BambooQA']['iter2']} | — | "
            f"{agg['BambooQA']['iter0']} | {agg['BambooQA']['iter1']} | {agg['BambooQA']['iter2']} | — | "
            f"{agg['BambooQA']['iter0']} | {agg['BambooQA']['iter1']} | {agg['BambooQA']['iter2']} | — |"
        )
        if old_bamboo in text:
            text = text.replace(old_bamboo, new_bamboo, 1)
        else:
            print("WARN: BambooQA RG-CTO row not found", file=sys.stderr)

    if "HMMT24" in agg and "HLE" in agg:
        old_math = (
            "| Phi-4-Reasoning | RG-CTO | 28.0 | — | — | — | — | — | 5.9 | — | — |"
        )
        new_math = (
            f"| Phi-4-Reasoning | RG-CTO | "
            f"{agg['HMMT24']['iter0']} | {agg['HMMT24']['iter1']} | {agg['HMMT24']['iter2']} | "
            f"— | — | — | "
            f"{agg['HLE']['iter0']} | {agg['HLE']['iter1']} | {agg['HLE']['iter2']} |"
        )
        if old_math in text:
            text = text.replace(old_math, new_math, 1)
        else:
            print("WARN: Phi-4 math RG-CTO row not found", file=sys.stderr)

    note_marker = "RG-CTO 待跑"
    if note_marker in text and len(agg) >= 2:
        text = text.replace(
            note_marker,
            "Phi-4 RG-CTO 三数据集 3-run 已完成（`results/runs/*_Phi_4_Reasoning_RG_CTO/`）",
            1,
        )

    md_file.write_text(text, encoding="utf-8")
    summary_path = project_root / "results/summaries/rg_cto_phi4_mean_std.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched {md_file}")
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
