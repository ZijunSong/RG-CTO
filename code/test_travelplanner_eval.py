#!/usr/bin/env python3
"""Checker tests for the TravelPlanner sole-planning subset."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_prompts import (  # noqa: E402
    get_baseline_system_prompt,
    get_cto_prefixes,
    get_distillation_prompt,
    get_experience_guided_system_prompt,
    resolve_task_type,
)
from travelplanner_eval import evaluate_text, parse_plan  # noqa: E402


def _aligned(columns, rows):
    widths = [
        max([len(column)] + [len(str(row[index])) for row in rows])
        for index, column in enumerate(columns)
    ]

    def line(values):
        return "  ".join(str(value).rjust(width) for value, width in zip(values, widths))

    return "\n".join([line(columns)] + [line(row) for row in rows])


def _toy_record():
    flights = _aligned(
        ["Flight Number", "Price", "DepTime", "ArrTime", "ActualElapsedTime", "FlightDate", "OriginCityName", "DestCityName", "Distance"],
        [
            ["F0000001", "40", "08:00", "09:00", "1 hours 0 minutes", "2022-03-01", "Alpha", "Beta", "100"],
            ["F0000002", "50", "18:00", "19:00", "1 hours 0 minutes", "2022-03-03", "Beta", "Alpha", "100"],
        ],
    )
    restaurants = _aligned(
        ["Name", "Average Cost", "Cuisines", "Aggregate Rating", "City"],
        [
            ["North Cafe", "10", "Cafe, French", "4.0", "Beta"],
            ["East Diner", "12", "American, Italian", "4.2", "Beta"],
            ["South Grill", "8", "Chinese, BBQ", "3.9", "Beta"],
        ],
    )
    attractions = _aligned(
        ["Name", "Latitude", "Longitude", "Address", "Phone", "Website", "City"],
        [["City Park", "1.0", "2.0", "1 Main St, Beta, CA 90001, USA", "000", "http://park.example", "Beta"]],
    )
    hotels = _aligned(
        ["NAME", "price", "room type", "house_rules", "minimum nights", "maximum occupancy", "review rate number", "city"],
        [["Quiet Room", "30.0", "Private room", "No parties", "1.0", "2", "4.0", "Beta"]],
    )
    reference = [
        {"Description": "Attractions in Beta", "Content": attractions},
        {"Description": "Restaurants in Beta", "Content": restaurants},
        {"Description": "Accommodations in Beta", "Content": hotels},
        {"Description": "Flight from Alpha to Beta on 2022-03-01", "Content": flights},
        {"Description": "Self-driving from Alpha to Beta", "Content": "self-driving, from Alpha to Beta, cost: 20"},
        {"Description": "Taxi from Alpha to Beta", "Content": "taxi, from Alpha to Beta, cost: 80"},
        {"Description": "Flight from Beta to Alpha on 2022-03-03", "Content": flights},
        {"Description": "Self-driving from Beta to Alpha", "Content": "self-driving, from Beta to Alpha, cost: 20"},
        {"Description": "Taxi from Beta to Alpha", "Content": "taxi, from Beta to Alpha, cost: 80"},
    ]
    return {
        "org": "Alpha",
        "dest": "Beta",
        "days": 3,
        "visiting_city_number": 1,
        "people_number": 2,
        "budget": 500,
        "local_constraint": {
            "house rule": None,
            "cuisine": ["French"],
            "room type": "private room",
            "transportation": "no self-driving",
        },
        "city_state": {"Alpha": "", "Beta": "California"},
        "reference_information": reference,
    }


def _good_plan():
    return [
        {
            "current_city": "from Alpha to Beta",
            "transportation": "Flight Number: F0000001, from Alpha to Beta, Departure Time: 08:00, Arrival Time: 09:00",
            "breakfast": "-",
            "lunch": "-",
            "dinner": "-",
            "attraction": "-",
            "accommodation": "Quiet Room, Beta",
        },
        {
            "current_city": "Beta",
            "transportation": "-",
            "breakfast": "North Cafe, Beta",
            "lunch": "East Diner, Beta",
            "dinner": "South Grill, Beta",
            "attraction": "City Park, Beta;",
            "accommodation": "Quiet Room, Beta",
        },
        {
            "current_city": "from Beta to Alpha",
            "transportation": "Flight Number: F0000002, from Beta to Alpha, Departure Time: 18:00, Arrival Time: 19:00",
            "breakfast": "-",
            "lunch": "-",
            "dinner": "-",
            "attraction": "-",
            "accommodation": "-",
        },
    ]


class TravelPlannerEvalTest(unittest.TestCase):
    def test_good_plan_passes(self):
        record = _toy_record()
        text = "```json\n" + json.dumps(_good_plan()) + "\n```"
        scored = evaluate_text(record, text)
        self.assertTrue(scored["delivered"])
        self.assertTrue(scored["final_pass"], scored["commonsense"] | (scored["hard"] or {}))

    def test_repeated_restaurant_fails(self):
        plan = _good_plan()
        plan[1]["lunch"] = "North Cafe, Beta"
        scored = evaluate_text(_toy_record(), json.dumps(plan))
        self.assertFalse(scored["commonsense"]["is_valid_restaurants"][0])
        self.assertFalse(scored["final_pass"])

    def test_invented_flight_fails_sandbox(self):
        plan = _good_plan()
        plan[0]["transportation"] = "Flight Number: F9999999, from Alpha to Beta, Departure Time: 08:00, Arrival Time: 09:00"
        scored = evaluate_text(_toy_record(), json.dumps(plan))
        self.assertFalse(scored["commonsense"]["is_valid_information_in_sandbox"][0])
        self.assertFalse(scored["final_pass"])

    def test_self_driving_violates_hard_constraint(self):
        plan = _good_plan()
        plan[0]["transportation"] = "Self-driving, from Alpha to Beta"
        plan[2]["transportation"] = "Self-driving, from Beta to Alpha"
        scored = evaluate_text(_toy_record(), json.dumps(plan))
        self.assertTrue(scored["commonsense"]["is_valid_information_in_sandbox"][0], scored["commonsense"])
        self.assertFalse(scored["hard"]["valid_transportation"][0])
        self.assertFalse(scored["final_pass"])

    def test_over_budget(self):
        record = _toy_record()
        record["budget"] = 1
        scored = evaluate_text(record, json.dumps(_good_plan()))
        self.assertFalse(scored["hard"]["valid_cost"][0])
        self.assertFalse(scored["final_pass"])

    def test_natural_language_plan(self):
        text = """
Day 1:
Current City: from Alpha to Beta
Transportation: Flight Number: F0000001, from Alpha to Beta, Departure Time: 08:00, Arrival Time: 09:00
Breakfast: -
Lunch: -
Dinner: -
Attraction: -
Accommodation: Quiet Room, Beta
Day 2:
Current City: Beta
Transportation: -
Breakfast: North Cafe, Beta
Lunch: East Diner, Beta
Dinner: South Grill, Beta
Attraction: City Park, Beta;
Accommodation: Quiet Room, Beta
Day 3:
Current City: from Beta to Alpha
Transportation: Flight Number: F0000002, from Beta to Alpha, Departure Time: 18:00, Arrival Time: 19:00
Breakfast: -
Lunch: -
Dinner: -
Attraction: -
Accommodation: -
"""
        self.assertEqual(len(parse_plan(text)), 3)
        self.assertTrue(evaluate_text(_toy_record(), text)["final_pass"])

    def test_empty_completion_is_not_delivered(self):
        scored = evaluate_text(_toy_record(), "I cannot plan this.")
        self.assertFalse(scored["delivered"])
        self.assertFalse(scored["final_pass"])
        self.assertEqual(scored["commonsense_passed"], 0)
        self.assertGreater(scored["hard_applicable"], 0)
        self.assertEqual(scored["hard_passed"], 0)

    def test_task_routing(self):
        self.assertEqual(resolve_task_type(dataset="TravelPlanner_Val60"), "agent")
        self.assertEqual(resolve_task_type(input_path="data/TravelPlanner_Val60.jsonl"), "agent")
        self.assertIn("JSON", get_baseline_system_prompt("agent"))
        guided = get_experience_guided_system_prompt("agent")
        self.assertIn("Experience Bank", guided)
        guided.format(experience_context="none")
        pos, neg, fallback = get_cto_prefixes("agent")
        self.assertIn("Propositions", pos)
        self.assertIn("dead ends", neg)
        self.assertIn("JSON", fallback)
        prompt = get_distillation_prompt("agent", "cf_exp")
        self.assertIn("{{success_divergence_fragment}}", prompt)
        self.assertIn("{{failure_minimal_head}}", get_distillation_prompt("agent", "cf_min_edit"))


class RealSubsetTest(unittest.TestCase):
    def test_first_validation_query_accepts_a_table_plan(self):
        path = Path(__file__).resolve().parents[1] / "data" / "TravelPlanner_Val60.jsonl"
        if not path.exists():
            self.skipTest("subset jsonl not built")
        with path.open(encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        self.assertEqual(record["question_type"], "agent")
        self.assertIn("Given information:", record["question"])
        self.assertEqual(record["days"], 3)
        from travelplanner_eval import Catalog

        catalog = Catalog(record["reference_information"])
        city = record["dest"]
        flights_out = [
            row for row in catalog.flights
            if row["OriginCityName"] == record["org"] and row["DestCityName"] == city
        ]
        flights_back = [
            row for row in catalog.flights
            if row["OriginCityName"] == city and row["DestCityName"] == record["org"]
        ]
        self.assertTrue(flights_out and flights_back)
        restaurants = sorted(
            (row for row in catalog.restaurants if row["City"] == city),
            key=lambda row: float(row["Average Cost"]),
        )[:3]
        attraction = next(row for row in catalog.attractions if row["City"] == city)
        people = int(record["people_number"])
        hotels = [
            row for row in catalog.accommodations
            if row["city"] == city and float(row["minimum nights"]) <= 2 and float(row["maximum occupancy"]) > 0
        ]
        self.assertTrue(hotels)
        hotel = min(
            hotels,
            key=lambda row: float(row["price"]) * ((people + float(row["maximum occupancy"]) - 1) // float(row["maximum occupancy"])),
        )
        flights_out.sort(key=lambda row: float(row["Price"]))
        flights_back.sort(key=lambda row: float(row["Price"]))
        out = flights_out[0]
        back = flights_back[0]
        plan = [
            {
                "current_city": f"from {record['org']} to {city}",
                "transportation": (
                    f"Flight Number: {out['Flight Number']}, from {record['org']} to {city}, "
                    f"Departure Time: {out['DepTime']}, Arrival Time: {out['ArrTime']}"
                ),
                "breakfast": "-",
                "lunch": "-",
                "dinner": "-",
                "attraction": "-",
                "accommodation": f"{hotel['NAME']}, {city}",
            },
            {
                "current_city": city,
                "transportation": "-",
                "breakfast": f"{restaurants[0]['Name']}, {city}",
                "lunch": f"{restaurants[1]['Name']}, {city}",
                "dinner": f"{restaurants[2]['Name']}, {city}",
                "attraction": f"{attraction['Name']}, {city};",
                "accommodation": f"{hotel['NAME']}, {city}",
            },
            {
                "current_city": f"from {city} to {record['org']}",
                "transportation": (
                    f"Flight Number: {back['Flight Number']}, from {city} to {record['org']}, "
                    f"Departure Time: {back['DepTime']}, Arrival Time: {back['ArrTime']}"
                ),
                "breakfast": "-",
                "lunch": "-",
                "dinner": "-",
                "attraction": "-",
                "accommodation": "-",
            },
        ]
        scored = evaluate_text(record, json.dumps(plan))
        self.assertTrue(scored["commonsense"]["is_valid_information_in_sandbox"][0], scored["commonsense"])
        self.assertTrue(scored["commonsense"]["is_reasonable_visiting_city"][0], scored["commonsense"])
        self.assertTrue(scored["commonsense"]["is_not_absent"][0], scored["commonsense"])
        self.assertTrue(scored["final_pass"], {"commonsense": scored["commonsense"], "hard": scored["hard"]})


if __name__ == "__main__":
    unittest.main()
