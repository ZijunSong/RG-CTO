#!/usr/bin/env python3
"""Build data/TripPlanning50.jsonl from the normalized NATURAL PLAN parquet."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "trip_planning_test.parquet"
OUTPUT = ROOT / "data" / "TripPlanning50.jsonl"
N = 50


def main() -> None:
    frame = pd.read_parquet(SOURCE).head(N)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for index, row in frame.iterrows():
            record = {
                "question_id": str(row["id"]),
                "question": str(row["prompt_5shot"]),
                "answer": str(row["golden_plan_text"]),
                "question_type": "agent",
                "checker": "trip_planning",
                "dataset": "TripPlanning50",
                "num_cities": int(row["num_cities"]),
                "cities": str(row["cities_raw"]),
                "durations": str(row["durations_raw"]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(frame)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
