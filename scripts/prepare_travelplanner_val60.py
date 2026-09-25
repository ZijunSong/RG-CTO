#!/usr/bin/env python3
"""Build data/TravelPlanner_Val60.jsonl from the official validation split.

Sampling is stratified and deterministic: validation has 9 groups
(easy/medium/hard x 3/5/7 days) of 20 queries. Easy and medium groups
contribute the first 7 queries; hard groups contribute the first 6.
That is 60 queries, smaller than CodeContests Test 165.

Source: https://huggingface.co/datasets/osunlp/TravelPlanner (validation.csv).
Sole-planning reference tables are already in that file.
"""
from __future__ import annotations

import ast
import csv
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "validation.csv"
OUT = ROOT / "data" / "TravelPlanner_Val60.jsonl"
CSV_URLS = (
    "https://huggingface.co/datasets/osunlp/TravelPlanner/resolve/main/validation.csv",
    "https://hf-mirror.com/datasets/osunlp/TravelPlanner/resolve/main/validation.csv",
)

sys.path.insert(0, str(ROOT / "code"))
from travelplanner_eval import Catalog  # noqa: E402


def _ensure_csv() -> None:
    if RAW.exists() and RAW.stat().st_size > 1_000_000:
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in CSV_URLS:
        print(f"Downloading {url}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                RAW.write_bytes(response.read())
            return
        except Exception as exc:
            last_error = exc
            print(f"  failed: {exc}")
    raise SystemExit(f"Could not download validation.csv: {last_error}")


def _load_rows():
    with RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 180:
        raise SystemExit(f"Expected 180 validation queries, found {len(rows)}")
    return rows


def _select(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["level"], int(row["days"]))].append(index)
    chosen = []
    for level in ("easy", "medium", "hard"):
        for days in (3, 5, 7):
            take = 7 if level in ("easy", "medium") else 6
            indexes = groups[(level, days)]
            if len(indexes) < take:
                raise SystemExit(f"Group {level}/{days} has only {len(indexes)} queries")
            chosen.extend(indexes[:take])
    if len(chosen) != 60:
        raise SystemExit(f"Expected 60 queries, selected {len(chosen)}")
    return chosen


def _format_reference(blocks) -> str:
    parts = []
    for block in blocks:
        parts.append(f"{block['Description']}:\n{block['Content'].strip()}")
    return "\n\n".join(parts)


def _question(query: str, blocks) -> str:
    return (
        "Given information:\n"
        f"{_format_reference(blocks)}\n\n"
        f"Query: {query}\n"
    )


def main() -> None:
    _ensure_csv()
    rows = _load_rows()
    global_state = {}
    parsed_blocks = []
    for row in rows:
        blocks = ast.literal_eval(row["reference_information"])
        parsed_blocks.append(blocks)
        global_state.update(Catalog(blocks).city_state)

    records = []
    for question_id, source_index in enumerate(_select(rows), start=1):
        row = rows[source_index]
        blocks = parsed_blocks[source_index]
        catalog = Catalog(blocks)
        city_state = dict(catalog.city_state)
        org = row["org"].strip()
        if org not in city_state:
            city_state[org] = global_state.get(org, "")
        records.append(
            {
                "question_id": question_id,
                "question": _question(row["query"], blocks),
                "answer": "final_pass",
                "question_type": "agent",
                "checker": "travelplanner",
                "dataset": "TravelPlanner_Val60",
                "travelplanner_index": source_index,
                "org": row["org"],
                "dest": row["dest"],
                "days": int(row["days"]),
                "visiting_city_number": int(row["visiting_city_number"]),
                "date": ast.literal_eval(row["date"]),
                "people_number": int(row["people_number"]),
                "local_constraint": ast.literal_eval(row["local_constraint"]),
                "budget": float(row["budget"]),
                "query": row["query"],
                "level": row["level"],
                "reference_information": blocks,
                "city_state": city_state,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} queries to {OUT}")


if __name__ == "__main__":
    main()
