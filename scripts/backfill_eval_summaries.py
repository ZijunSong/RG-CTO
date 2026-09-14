#!/usr/bin/env python3
"""Regenerate run*_eval_summary.json from existing step pass_at_1.json files."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_summary(out_prefix: Path, model_name: str | None = None) -> dict:
    for it, step in [(0, 1), (1, 3), (2, 5)]:
        ver_dir = Path(f"{out_prefix}_step{step}") / "results"
        if not ver_dir.is_dir():
            continue
        if model_name and not (ver_dir / "pass_at_1.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "eval/calculate_pass_at_k_from_completions.py",
                    "--verification_dir",
                    str(ver_dir),
                    "--k_values",
                    "1",
                    "--output_file",
                    str(ver_dir / "pass_at_1.json"),
                    "--max_reference",
                    "32",
                    "--tokenizer_path",
                    model_name,
                ],
                check=False,
                cwd=Path(__file__).resolve().parent.parent,
            )

    summary: dict[str, float | None] = {}
    for it, step in [(0, 1), (1, 3), (2, 5)]:
        f = Path(f"{out_prefix}_step{step}") / "results" / "pass_at_1.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        v = d.get("pass_at_k", {}).get("pass@1")
        if v is None:
            v = d.get("pass@1")
        summary[f"iter{it}"] = round(float(v) * 100, 2) if v is not None else None

    out_json = out_prefix.parent / f"{out_prefix.name}_eval_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    runs_root = project_root / "results" / "runs"
    models = {
        "Phi_4": "/data/ppnm/models/Phi-4-reasoning",
        "Qwen3_30B": "/data/ppnm/models/Qwen3-30B-A3B-Thinking-2507",
    }

    targets = [
        ("HMMT25_Qwen3_30B_A3B_Thinking_2507_RSE", models["Qwen3_30B"]),
        ("HMMT24_Qwen3_30B_A3B_Thinking_2507_RG_CTO", models["Qwen3_30B"]),
        ("HMMT25_Qwen3_30B_A3B_Thinking_2507_RG_CTO", models["Qwen3_30B"]),
        ("HLE_math_text_Qwen3_30B_A3B_Thinking_2507_RG_CTO", models["Qwen3_30B"]),
        ("BambooQA_Phi_4_Reasoning_RG_CTO", models["Phi_4"]),
        ("BambooQA_Phi_4_Reasoning_RG_CTO_gpu1", models["Phi_4"]),
        ("BambooQA_Phi_4_Reasoning_RG_CTO_gpu2", models["Phi_4"]),
    ]

    for name, model in targets:
        base = runs_root / name
        if not base.is_dir():
            continue
        for run_dir in sorted(base.glob("run[0-9]_step1")):
            run_id = run_dir.name.replace("_step1", "")
            out_prefix = base / run_id
            summary = write_summary(out_prefix, model)
            print(f"{name}/{run_id}: {summary}")


if __name__ == "__main__":
    main()
