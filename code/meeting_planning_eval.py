"""Meeting Planning checker, adapted from NATURAL PLAN.

The decision rule matches google-deepmind/natural-plan
evaluate_meeting_planning.py: a plan is correct when the number of valid
meetings equals the number accepted in the golden plan. Travel time comes
from the example's distance matrix, not from the minutes written in the text.
"""
from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def convert_to_time_obj(time_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(time_str.strip(), "%I:%M%p")


def process_constraints(data: Sequence[Sequence[Any]]) -> Dict[str, Dict[str, Any]]:
    constraints: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for name, location, times, meeting_time in data:
        constraints[name]["location"] = location
        start_time = convert_to_time_obj(str(times).split("to")[0].strip())
        end_time = convert_to_time_obj(str(times).split("to")[1].strip())
        constraints[name]["start_time"] = start_time
        constraints[name]["end_time"] = end_time
        constraints[name]["meeting_time"] = int(meeting_time)
    return constraints


def parse_text_plan(plan: str) -> List[str]:
    """Split a model solution into the official sentence steps."""
    if not plan:
        return []
    if "</think>" in plan:
        plan = plan.split("</think>", 1)[-1]
    prefix = "SOLUTION:"
    if prefix in plan:
        plan = plan[plan.find(prefix) + len(prefix):].strip()
    steps = []
    for step in plan.split("."):
        step = step.strip()
        if step:
            steps.append(step)
    return steps


def validator_from_text(
    plan: Sequence[str],
    processed_constraints: Dict[str, Any],
    start_location: str,
    initial_time: str,
    dist_matrix: Dict[str, Any],
) -> int:
    """Return how many meetings are valid before the first illegal step."""
    met_with = {}
    score = 0
    cur_location = start_location
    cur_time = convert_to_time_obj(initial_time)
    for step in plan:
        try:
            if step.startswith("You start"):
                continue
            if step.startswith("You travel"):
                destination = step.split("travel to ", 1)[1].split(" in", 1)[0].strip()
                cur_time = cur_time + datetime.timedelta(
                    minutes=int(dist_matrix[cur_location][destination])
                )
                cur_location = destination
            elif step.startswith("You wait"):
                raw_end_time = step.split("wait until ", 1)[1].split(".", 1)[0].strip()
                end_time = convert_to_time_obj(raw_end_time)
                if end_time <= cur_time:
                    break
                cur_time = end_time
            elif step.startswith("You meet"):
                person = step.split("meet ", 1)[1].split(" for", 1)[0].strip()
                if person in met_with:
                    break
                met_with[person] = 1
                new_time = cur_time + datetime.timedelta(
                    minutes=int(processed_constraints[person]["meeting_time"])
                )
                if (
                    cur_location == processed_constraints[person]["location"]
                    and cur_time >= processed_constraints[person]["start_time"]
                    and new_time <= processed_constraints[person]["end_time"]
                ):
                    score += 1
                    cur_time = new_time
                else:
                    break
            else:
                break
        except (KeyError, ValueError, IndexError):
            break
    return score


def _as_step_list(golden_plan: Any) -> List[str]:
    if isinstance(golden_plan, str):
        return parse_text_plan(golden_plan)
    return [str(step).strip().rstrip(".") for step in golden_plan if str(step).strip()]


def evaluate_text(record: Dict[str, Any], text: str) -> Dict[str, Any]:
    start_location, initial_time = record["constraints"][0]
    constraints = process_constraints(record["constraints"][1:])
    dist_matrix = record["dist_matrix"]
    golden_steps = _as_step_list(record.get("golden_plan") or record.get("answer") or [])
    golden_score = validator_from_text(
        golden_steps, constraints, start_location, initial_time, dist_matrix
    )
    pred_steps = parse_text_plan(text or "")
    pred_score = validator_from_text(
        pred_steps, constraints, start_location, initial_time, dist_matrix
    )
    delivered = pred_score > 0 or any(step.startswith("You start") for step in pred_steps)
    return {
        "delivered": delivered,
        "final_pass": pred_score == golden_score and golden_score > 0,
        "pred_score": pred_score,
        "golden_score": golden_score,
        "commonsense_passed": pred_score,
        "commonsense_total": golden_score,
        "hard_passed": pred_score,
        "hard_applicable": golden_score,
    }
