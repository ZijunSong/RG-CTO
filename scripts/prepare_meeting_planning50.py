#!/usr/bin/env python3
"""Write data/MeetingPlanning50.jsonl from NATURAL PLAN Meeting Planning.

Takes the first 50 examples in official order (meeting_planning_example_0
through 49). Each record keeps the 5-shot prompt, constraints, distance
matrix, and golden plan so the official validator can score a completion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "raw" / "meeting_planning_test.parquet"
OUT = ROOT / "data" / "MeetingPlanning50.jsonl"
N = 50


def _steps(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(step).strip() for step in list(value) if str(step).strip()]


def main() -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to read the parquet. Use the cto environment."
        ) from exc

    if not PARQUET.exists():
        raise SystemExit(f"Missing {PARQUET}. Download the NATURAL PLAN parquet first.")

    frame = pd.read_parquet(PARQUET)
    if len(frame) < N:
        raise SystemExit(f"Expected at least {N} rows, found {len(frame)}")
    slice_ = frame.iloc[:N]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for _, row in slice_.iterrows():
            constraints = json.loads(row["constraints_json"])
            record = {
                "question_id": row["id"],
                "question": row["prompt_5shot"],
                "answer": row["golden_plan_text"],
                "question_type": "agent",
                "checker": "meeting_planning",
                "dataset": "MeetingPlanning50",
                "num_people": int(row["num_people"]),
                "constraints": constraints,
                "dist_matrix": json.loads(row["dist_matrix_json"]),
                "golden_plan": _steps(row["golden_plan_steps"]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {N} examples to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
