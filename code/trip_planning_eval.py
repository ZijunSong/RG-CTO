"""NATURAL PLAN Trip Planning checker.

Port of the official parser in
google-deepmind/natural-plan ``evaluate_trip_planning.py``: a response is
correct only when the parsed city order and stay lengths exactly match the
golden plan.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


def _answer_text(text: str) -> str:
    text = text or ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "SOLUTION:" in text:
        text = text.split("SOLUTION:")[-1]
    return text.strip()


def parse_response(response: str) -> List[Tuple[str, int]]:
    pattern_visit = r"\d+-\d+"
    pattern_flight = r".*Day (\d+).*from (\w+) to (\w+)"
    pattern_days = r"European cities for (\d+) days"

    days, flights, flight_days = [], [], []
    total_days = None
    for piece in _answer_text(response).split("\n"):
        days_match = re.findall(pattern_days, piece)
        if days_match:
            total_days = int(days_match[0])

        visit_match = re.findall(pattern_visit, piece)
        if visit_match:
            days.append(visit_match[0])
            end_day = int(visit_match[0].split("-")[1])
            if total_days is not None and end_day == total_days:
                break

        flight_match = re.findall(pattern_flight, piece)
        if flight_match:
            flights.append(flight_match[0])

    if not days or not flights:
        return []

    visit_cities, parsed_flights = [], []
    for flight_day, city_from, city_to in flights:
        flight_days.append(int(flight_day))
        parsed_flights.append((city_from, city_to))

    visit_cities.append(parsed_flights[0][0])
    for _, city_to in parsed_flights:
        visit_cities.append(city_to)

    if total_days is None:
        return []
    flight_days = [1] + flight_days + [total_days]
    stays = []
    for index, city in enumerate(visit_cities):
        stays.append((city, flight_days[index + 1] - flight_days[index] + 1))
    return stays


def exact_match(cities: str, durations: str, response: str) -> bool:
    stays = parse_response(response)
    if not stays:
        return False
    parsed_plan = " ".join(f"{city} {days}" for city, days in stays)
    gold_cities = str(cities).split("**")
    gold_days = str(durations).split("**")
    gold_plan = " ".join(f"{city} {days}" for city, days in zip(gold_cities, gold_days))
    return parsed_plan == gold_plan


def evaluate_text(record: Dict, text: str) -> Dict:
    cities = record.get("cities") or record.get("answer") or ""
    durations = record.get("durations") or ""
    if isinstance(cities, Sequence) and not isinstance(cities, str):
        cities = "**".join(str(city) for city in cities)
    if isinstance(durations, Sequence) and not isinstance(durations, str):
        durations = "**".join(str(days) for days in durations)
    passed = exact_match(str(cities), str(durations), text or "")
    return {
        "delivered": bool(parse_response(text or "")),
        "final_pass": passed,
        "commonsense_passed": int(passed),
        "commonsense_total": 1,
        "hard_passed": int(passed),
        "hard_applicable": 1,
    }
