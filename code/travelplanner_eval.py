#!/usr/bin/env python3
"""Sole-planning checker for the TravelPlanner validation subset.

The official evaluator looks entities up in the full sandbox database. In
sole-planning mode that sandbox is exactly the reference information shipped
with each query, so this module parses those tables and applies the same
commonsense and hard constraints (Xie et al., 2024). A rollout is correct
only when every commonsense constraint passes and every applicable hard
constraint passes.
"""
from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

US_STATE_ABBR = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}

ATTRACTION_COLS = ["Name", "Latitude", "Longitude", "Address", "Phone", "Website", "City"]
RESTAURANT_COLS = ["Name", "Average Cost", "Cuisines", "Aggregate Rating", "City"]
ACCOMMODATION_COLS = [
    "NAME",
    "price",
    "room type",
    "house_rules",
    "minimum nights",
    "maximum occupancy",
    "review rate number",
    "city",
]
FLIGHT_COLS = [
    "Flight Number",
    "Price",
    "DepTime",
    "ArrTime",
    "ActualElapsedTime",
    "FlightDate",
    "OriginCityName",
    "DestCityName",
    "Distance",
]

COMMONSENSE_KEYS = (
    "is_reasonable_visiting_city",
    "is_valid_restaurants",
    "is_valid_attractions",
    "is_valid_accommodation",
    "is_valid_transportation",
    "is_valid_information_in_current_city",
    "is_valid_information_in_sandbox",
    "is_not_absent",
)

Check = Tuple[bool, Optional[str]]


def _as_obj(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return ast.literal_eval(text)
        if text[0] in "\"'":
            return ast.literal_eval(text)
    return value


def _column_edges(header: str, columns: Sequence[str]) -> List[int]:
    edges: List[int] = []
    pos = 0
    for name in columns:
        found = header.find(name, pos)
        if found < 0:
            raise ValueError(f"Missing column {name!r} in header {header!r}")
        edges.append(found + len(name))
        pos = found + len(name)
    return edges


def _parse_aligned_table(content: str, columns: Sequence[str]) -> List[Dict[str, str]]:
    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    edges = _column_edges(lines[0], columns)
    records = []
    for line in lines[1:]:
        values = []
        prev = 0
        for edge in edges:
            values.append(line[prev:edge].strip() if prev < len(line) else "")
            prev = edge
        records.append(dict(zip(columns, values)))
    return records


def _state_from_address(address: str) -> Optional[str]:
    match = re.search(r",\s*([A-Z]{2})\s+\d{5}", address or "")
    if not match:
        return None
    return US_STATE_ABBR.get(match.group(1), match.group(1))


class Catalog:
    """Entities and prices the sole-planning prompt is allowed to use."""

    def __init__(self, reference_information: Any):
        blocks = _as_obj(reference_information)
        if not isinstance(blocks, list):
            raise ValueError("reference_information must be a list of description/content blocks")
        self.attractions: List[Dict[str, str]] = []
        self.restaurants: List[Dict[str, str]] = []
        self.accommodations: List[Dict[str, str]] = []
        self.flights: List[Dict[str, str]] = []
        self.drives: Dict[Tuple[str, str, str], Optional[int]] = {}
        self.city_state: Dict[str, str] = {}
        city_state_votes: Dict[str, Dict[str, int]] = {}

        for block in blocks:
            description = str(block.get("Description", ""))
            content = str(block.get("Content", ""))
            if description.startswith("Attractions"):
                rows = _parse_aligned_table(content, ATTRACTION_COLS)
                self.attractions.extend(rows)
                for row in rows:
                    state = _state_from_address(row.get("Address", ""))
                    city = row.get("City", "").strip()
                    if state and city:
                        city_state_votes.setdefault(city, {})
                        city_state_votes[city][state] = city_state_votes[city].get(state, 0) + 1
            elif description.startswith("Restaurants"):
                self.restaurants.extend(_parse_aligned_table(content, RESTAURANT_COLS))
            elif description.startswith("Accommodations"):
                self.accommodations.extend(_parse_aligned_table(content, ACCOMMODATION_COLS))
            elif description.startswith("Flight"):
                if "there is no flight" in content.lower():
                    continue
                self.flights.extend(_parse_aligned_table(content, FLIGHT_COLS))
            elif description.startswith("Self-driving") or description.startswith("Taxi"):
                endpoints = re.match(r"^(Self-driving|Taxi) from (.+) to (.+)$", description)
                if not endpoints:
                    continue
                mode = "self-driving" if endpoints.group(1) == "Self-driving" else "taxi"
                origin = endpoints.group(2).strip()
                dest = endpoints.group(3).strip()
                if "no valid" in content.lower():
                    cost = None
                else:
                    cost_match = re.search(r"cost:\s*([\d,]+)", content, flags=re.I)
                    cost = int(cost_match.group(1).replace(",", "")) if cost_match else None
                self.drives[(mode, origin, dest)] = cost

        for city, votes in city_state_votes.items():
            self.city_state[city] = max(votes.items(), key=lambda item: item[1])[0]


def extract_before_parenthesis(text: Optional[str]) -> str:
    if text is None:
        return ""
    match = re.search(r"^(.*?)\([^)]*\)", text)
    return match.group(1) if match else text


def extract_from_to(text: str) -> Tuple[Optional[str], Optional[str]]:
    match = re.search(r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)", text or "")
    if not match:
        return None, None
    return match.group(1), match.group(2)


def get_valid_name_city(info: str) -> Tuple[str, str]:
    match = re.search(r"(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$", info or "")
    if not match:
        return "-", "-"
    return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip()).strip()


def _clean_city(city: Optional[str]) -> str:
    return extract_before_parenthesis(city or "").strip()


def _field(day: Dict[str, Any], key: str) -> str:
    value = day.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _normalize_day(raw: Dict[str, Any]) -> Dict[str, str]:
    day: Dict[str, str] = {}
    for key, value in raw.items():
        norm = re.sub(r"[\s\-]+", "_", str(key).strip().lower())
        day[norm] = "" if value is None else str(value).strip()
    return day


def _json_candidates(text: str) -> Iterable[str]:
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S):
        yield match.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        yield text[start : end + 1]


def _as_day_list(payload: Any) -> Optional[List[Dict[str, str]]]:
    if isinstance(payload, dict):
        if "plan" in payload:
            return _as_day_list(payload["plan"])
        if "current_city" in _normalize_day(payload) or "current city" in payload:
            return [_normalize_day(payload)]
        return None
    if not isinstance(payload, list) or not payload:
        return None
    if not all(isinstance(item, dict) for item in payload):
        return None
    days = [_normalize_day(item) for item in payload]
    if not any(day.get("current_city") for day in days):
        return None
    return days


def _parse_json_plan(text: str) -> Optional[List[Dict[str, str]]]:
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        days = _as_day_list(payload)
        if days:
            return days
    return None


def _parse_nl_plan(text: str) -> Optional[List[Dict[str, str]]]:
    chunks = re.split(r"(?im)^(?=Day\s+\d+\s*:)", text)
    days: List[Dict[str, str]] = []
    field_re = re.compile(
        r"(?i)^\s*(Current City|Transportation|Breakfast|Lunch|Dinner|Attraction|Accommodation)\s*:\s*(.*)$"
    )
    for chunk in chunks:
        if not re.match(r"(?i)Day\s+\d+", chunk.strip()):
            continue
        day: Dict[str, str] = {}
        for line in chunk.splitlines():
            match = field_re.match(line)
            if not match:
                continue
            key = re.sub(r"[\s\-]+", "_", match.group(1).strip().lower())
            day[key] = match.group(2).strip()
        if day.get("current_city"):
            days.append(day)
    return days or None


def parse_plan(text: str) -> Optional[List[Dict[str, str]]]:
    """Extract a day-by-day plan from a model completion."""
    if not text or not str(text).strip():
        return None
    segments = []
    if "</think>" in text:
        segments.append(text.split("</think>", 1)[-1])
    segments.append(text)
    for segment in segments:
        plan = _parse_json_plan(segment) or _parse_nl_plan(segment)
        if plan:
            return plan
    return None


def _attraction_items(value: str) -> List[str]:
    if not value or value == "-":
        return []
    parts = [part.strip().rstrip(".") for part in value.split(";")]
    return [part for part in parts if part and part != "-"]


def _lookup_name_city(
    rows: Sequence[Dict[str, str]],
    name_key: str,
    city_key: str,
    name: str,
    city: str,
) -> List[Dict[str, str]]:
    if not name or name == "-" or not city or city == "-":
        return []
    return [row for row in rows if name in str(row.get(name_key, "")) and str(row.get(city_key, "")).strip() == city]


def _count_runs(values: Sequence[str]) -> List[Tuple[str, int]]:
    if not values:
        return []
    runs: List[Tuple[str, int]] = []
    current = values[0]
    count = 1
    for value in values[1:]:
        if value == current:
            count += 1
        else:
            runs.append((current, count))
            current = value
            count = 1
    runs.append((current, count))
    return runs


def _is_valid_city_sequence(city_list: Sequence[str]) -> bool:
    if len(city_list) < 3:
        return False
    visited = set()
    index = 0
    while index < len(city_list):
        city = city_list[index]
        if city in visited and index not in (0, len(city_list) - 1):
            return False
        count = 0
        while index < len(city_list) and city_list[index] == city:
            count += 1
            index += 1
        if count == 1 and 0 < index - 1 < len(city_list) - 1:
            return False
        visited.add(city)
    return True


def _transportation_kind(text: str) -> Optional[str]:
    lowered = text.lower()
    if "taxi" in lowered:
        return "Taxi"
    if "self-driving" in lowered:
        return "Self-driving"
    if "flight" in lowered:
        return "Flight"
    return None


def _day_cities(day: Dict[str, str]) -> List[str]:
    current = _field(day, "current_city")
    if "from" in current:
        origin, dest = extract_from_to(current)
        return [_clean_city(origin), _clean_city(dest)]
    return [_clean_city(current)]


def _query_view(record: Dict[str, Any]) -> Dict[str, Any]:
    constraint = _as_obj(record.get("local_constraint") or {})
    if not isinstance(constraint, dict):
        constraint = {}
    city_state = record.get("city_state") or {}
    if isinstance(city_state, str):
        city_state = _as_obj(city_state)
    return {
        "org": str(record.get("org", "")).strip(),
        "dest": str(record.get("dest", "")).strip(),
        "days": int(record["days"]),
        "visiting_city_number": int(record["visiting_city_number"]),
        "people_number": int(record["people_number"]),
        "budget": float(record["budget"]),
        "local_constraint": constraint,
        "city_state": dict(city_state),
    }


def is_reasonable_visiting_city(query: Dict[str, Any], plan: Sequence[Dict[str, str]]) -> Check:
    city_list: List[str] = []
    limit = min(query["days"], len(plan))
    for index in range(limit):
        current = _field(plan[index], "current_city")
        if "from" in current:
            origin, dest = extract_from_to(current)
            origin = _clean_city(origin)
            dest = _clean_city(dest)
            if index == 0 and origin != query["org"]:
                return False, f"The first day's city should be {query['org']}."
            city_list.extend([origin, dest])
        else:
            city_list.append(_clean_city(current))
    if not city_list or city_list[0] != city_list[-1]:
        return False, "The trip should be a closed circle."
    if not _is_valid_city_sequence(city_list):
        return False, "The city sequence is invalid."
    city_state = query["city_state"]
    for index, city in enumerate(city_list):
        if city not in city_state:
            return False, f"{city} is not a valid city."
        if index not in (0, len(city_list) - 1) and query["days"] > 3 and city_state[city] != query["dest"]:
            return False, f"{city} is not in {query['dest']}."
    return True, None


def is_valid_restaurants(query: Dict[str, Any], plan: Sequence[Dict[str, str]]) -> Check:
    seen: List[str] = []
    for index in range(min(query["days"], len(plan))):
        for meal in ("breakfast", "lunch", "dinner"):
            value = _field(plan[index], meal)
            if not value or value == "-":
                continue
            if value in seen:
                return False, f"The restaurant in day {index + 1} {meal} is repeated."
            seen.append(value)
    return True, None


def is_valid_attractions(query: Dict[str, Any], plan: Sequence[Dict[str, str]]) -> Check:
    seen: List[str] = []
    for index in range(min(query["days"], len(plan))):
        for attraction in _attraction_items(_field(plan[index], "attraction")):
            if attraction in seen:
                return False, f"The attraction '{attraction}' in day {index + 1} is repeated."
            seen.append(attraction)
    return True, None


def is_valid_accommodation(query: Dict[str, Any], plan: Sequence[Dict[str, str]], catalog: Catalog) -> Check:
    stays = []
    for index in range(min(query["days"], len(plan))):
        if "accommodation" not in plan[index]:
            return False, "No Accommodation Info."
        stays.append(_field(plan[index], "accommodation"))
    for value, nights in _count_runs(stays):
        if not value or value == "-":
            continue
        name, city = get_valid_name_city(value)
        hits = _lookup_name_city(catalog.accommodations, "NAME", "city", name, city)
        if len(hits) == 1:
            minimum = float(hits[0]["minimum nights"])
            if nights < minimum:
                return False, f"The accommodation {value} do not obey the minumum nights rule."
    return True, None


def is_valid_transportation(query: Dict[str, Any], plan: Sequence[Dict[str, str]]) -> Check:
    if not plan:
        return False, "The transportation in day 1 should not be empty."
    first = _field(plan[0], "transportation")
    if not first or first == "-":
        return False, "The transportation in day 1 should not be empty."
    kinds = [_transportation_kind(first)]
    for index in range(min(query["days"], len(plan))):
        value = _field(plan[index], "transportation")
        if value and value != "-":
            kinds.append(_transportation_kind(value))
    if ("Self-driving" in kinds and "Flight" in kinds) or ("Taxi" in kinds and "Self-driving" in kinds):
        return False, "The transportation is conflicting."
    return True, None


def is_valid_information_in_current_city(query: Dict[str, Any], plan: Sequence[Dict[str, str]]) -> Check:
    for index in range(min(query["days"], len(plan))):
        day = plan[index]
        cities = [city for city in _day_cities(day) if city]
        transport = _field(day, "transportation")
        if transport and transport != "-":
            for city in cities:
                if city not in transport:
                    return False, f"The transportation in day {index + 1} is invalid city choice."
        for meal in ("breakfast", "lunch", "dinner"):
            value = _field(day, meal)
            if value and value != "-" and not any(city in value for city in cities):
                return False, f"The {meal} in day {index + 1} is invalid city choice."
        for attraction in _attraction_items(_field(day, "attraction")):
            if not any(city in attraction for city in cities):
                return False, f"The attraction in day {index + 1} is invalid city choice."
        stay = _field(day, "accommodation")
        if stay and stay != "-" and cities and cities[-1] not in stay:
            return False, f"The accommodation in day {index + 1} is invalid city choice."
    return True, None


def is_valid_information_in_sandbox(query: Dict[str, Any], plan: Sequence[Dict[str, str]], catalog: Catalog) -> Check:
    for index in range(min(query["days"], len(plan))):
        day = plan[index]
        transport = _field(day, "transportation")
        if transport and transport != "-":
            origin, dest = extract_from_to(transport)
            if origin is None or dest is None:
                origin, dest = extract_from_to(_field(day, "current_city"))
            origin = _clean_city(origin)
            dest = _clean_city(dest)
            lowered = transport.lower()
            if "flight number" in lowered:
                try:
                    flight_number = transport.split("Flight Number: ", 1)[1].split(",", 1)[0].strip()
                except IndexError:
                    return False, f"The flight number in day {index + 1} is invalid in the sandbox."
                hits = [
                    row
                    for row in catalog.flights
                    if row.get("Flight Number", "").strip() == flight_number
                    and row.get("OriginCityName", "").strip() == origin
                    and row.get("DestCityName", "").strip() == dest
                ]
                if not hits:
                    return False, f"The flight number in day {index + 1} is invalid in the sandbox."
            elif "self-driving" in lowered or "taxi" in lowered:
                mode = "self-driving" if "self-driving" in lowered else "taxi"
                key = (mode, origin, dest)
                if catalog.drives.get(key) is None:
                    return False, f"The {mode} in day {index + 1} is invalid in the sandbox."
        for meal in ("breakfast", "lunch", "dinner"):
            value = _field(day, meal)
            if not value or value == "-":
                continue
            name, city = get_valid_name_city(value)
            if not _lookup_name_city(catalog.restaurants, "Name", "City", name, city):
                return False, f"The {meal} in day {index + 1} is invalid in the sandbox."
        for attraction in _attraction_items(_field(day, "attraction")):
            name, city = get_valid_name_city(attraction)
            if not _lookup_name_city(catalog.attractions, "Name", "City", name, city):
                return False, f"The attraction {attraction} in day {index + 1} is invalid in the sandbox."
        stay = _field(day, "accommodation")
        if stay and stay != "-":
            name, city = get_valid_name_city(stay)
            if not _lookup_name_city(catalog.accommodations, "NAME", "city", name, city):
                return False, f"The accommodation in day {index + 1} is invalid in the sandbox."
    return True, None


def _filled_day_count(query: Dict[str, Any], plan: Sequence[Dict[str, str]]) -> int:
    count = 0
    placeholder = "You don't need to fill in the information for this or later days."
    for index in range(min(query["days"], len(plan))):
        current = _field(plan[index], "current_city")
        if plan[index] and current != placeholder:
            count += 1
    return count


def is_valid_days(query: Dict[str, Any], plan: Sequence[Dict[str, str]]) -> Check:
    if _filled_day_count(query, plan) != query["days"]:
        return False, f"The number of days should be {query['days']}."
    return True, None


def is_valid_visiting_city_number(query: Dict[str, Any], plan: Sequence[Dict[str, str]]) -> Check:
    cities = set()
    for index in range(min(query["days"], len(plan))):
        current = _field(plan[index], "current_city")
        if "from" in current:
            origin, dest = extract_from_to(current)
            origin = _clean_city(origin)
            dest = _clean_city(dest)
            if index == 0 and origin != query["org"]:
                return False, f"The first day's city should be {query['org']}."
            cities.add(origin)
            cities.add(dest)
        else:
            cities.add(_clean_city(current))
    cities.discard(query["org"])
    if len(cities) != query["visiting_city_number"]:
        return False, f"The number of visiting cities should be {query['visiting_city_number']}."
    return True, None


def is_not_absent(query: Dict[str, Any], plan: Sequence[Dict[str, str]]) -> Check:
    if not is_valid_days(query, plan)[0]:
        return False, "Invalid Days"
    if not is_valid_visiting_city_number(query, plan)[0]:
        return False, "Invalid City Number"
    needed = 6 * query["days"]
    filled = 0
    for index in range(min(query["days"], len(plan))):
        day = plan[index]
        for key in ("transportation", "breakfast", "lunch", "dinner", "attraction", "accommodation"):
            if key not in day:
                return False, f"No {key} Info."
        current = _field(day, "current_city")
        traveling = "from " in current or " to " in current
        if traveling and _field(day, "transportation") in ("", "-"):
            return False, f"No transportation in day {index + 1} is not allowed."
        if not traveling and _field(day, "attraction") in ("", "-"):
            return False, f"No attaction in day {index + 1} is not allowed."
        if index != query["days"] - 1 and _field(day, "accommodation") in ("", "-"):
            return False, f"No accommodation in day {index + 1} is not allowed."
        if "from " not in current and any(_field(day, meal) in ("", "-") for meal in ("breakfast", "lunch", "dinner")):
            return False, f"No meal in day {index + 1} is not allowed."
        for value in day.values():
            if value and value != "-":
                filled += 1
    if filled / needed < 0.5:
        return False, "The absent information is more than 50%."
    return True, None


def _flight_price(catalog: Catalog, transport: str, origin: str, dest: str) -> Optional[float]:
    try:
        flight_number = transport.split("Flight Number: ", 1)[1].split(",", 1)[0].strip()
    except IndexError:
        return None
    hits = [row for row in catalog.flights if row.get("Flight Number", "").strip() == flight_number]
    if not hits:
        return None
    return float(hits[0]["Price"])


def get_total_cost(query: Dict[str, Any], plan: Sequence[Dict[str, str]], catalog: Catalog) -> float:
    total = 0.0
    people = query["people_number"]
    for index in range(min(query["days"], len(plan))):
        day = plan[index]
        transport = _field(day, "transportation")
        if transport and transport != "-":
            origin, dest = extract_from_to(transport)
            if origin is None or dest is None:
                origin, dest = extract_from_to(_field(day, "current_city"))
            origin = _clean_city(origin)
            dest = _clean_city(dest)
            lowered = transport.lower()
            if "flight number" in lowered:
                price = _flight_price(catalog, transport, origin, dest)
                if price is not None:
                    total += price * people
            elif "self-driving" in lowered or "taxi" in lowered:
                mode = "self-driving" if "self-driving" in lowered else "taxi"
                cost = catalog.drives.get((mode, origin, dest))
                if cost is not None:
                    capacity = 5 if mode == "self-driving" else 4
                    total += cost * math.ceil(people / capacity)
        for meal in ("breakfast", "lunch", "dinner"):
            value = _field(day, meal)
            if not value or value == "-":
                continue
            name, city = get_valid_name_city(value)
            hits = _lookup_name_city(catalog.restaurants, "Name", "City", name, city)
            if hits:
                total += float(hits[0]["Average Cost"]) * people
        stay = _field(day, "accommodation")
        if stay and stay != "-":
            name, city = get_valid_name_city(stay)
            hits = _lookup_name_city(catalog.accommodations, "NAME", "city", name, city)
            if hits:
                occupancy = float(hits[0]["maximum occupancy"])
                total += float(hits[0]["price"]) * math.ceil(people / occupancy)
    return total


def _hard_room_rule(query: Dict[str, Any], plan: Sequence[Dict[str, str]], catalog: Catalog) -> Tuple[Optional[bool], Optional[str]]:
    rule = query["local_constraint"].get("house rule")
    if not rule:
        return None, None
    forbidden = {
        "smoking": "No smoking",
        "parties": "No parties",
        "children under 10": "No children under 10",
        "visitors": "No visitors",
        "pets": "No pets",
    }.get(rule)
    for index in range(min(query["days"], len(plan))):
        stay = _field(plan[index], "accommodation")
        if not stay or stay == "-":
            continue
        name, city = get_valid_name_city(stay)
        hits = _lookup_name_city(catalog.accommodations, "NAME", "city", name, city)
        if hits and forbidden and forbidden in str(hits[0].get("house_rules", "")):
            return False, f"The house rule should be {rule}."
    return True, None


def _hard_cuisine(query: Dict[str, Any], plan: Sequence[Dict[str, str]], catalog: Catalog) -> Tuple[Optional[bool], Optional[str]]:
    required = query["local_constraint"].get("cuisine")
    if not required:
        return None, None
    found = set()
    for index in range(min(query["days"], len(plan))):
        for meal in ("breakfast", "lunch", "dinner"):
            value = _field(plan[index], meal)
            if not value or value == "-":
                continue
            name, city = get_valid_name_city(value)
            if city == query["org"]:
                continue
            hits = _lookup_name_city(catalog.restaurants, "Name", "City", name, city)
            if not hits:
                continue
            cuisines = str(hits[0].get("Cuisines", ""))
            for cuisine in required:
                if cuisine in cuisines:
                    found.add(cuisine)
    missing = [cuisine for cuisine in required if cuisine not in found]
    if missing:
        return False, f"The cuisine {missing[0]} is not satisfied."
    return True, None


def _hard_transport(query: Dict[str, Any], plan: Sequence[Dict[str, str]]) -> Tuple[Optional[bool], Optional[str]]:
    rule = query["local_constraint"].get("transportation")
    if not rule:
        return None, None
    for index in range(min(query["days"], len(plan))):
        value = _field(plan[index], "transportation")
        if not value or value == "-":
            continue
        if rule == "no flight" and "Flight" in value:
            return False, f"The transportation should not be {rule}."
        if rule == "no self-driving" and "Self-driving" in value:
            return False, f"The transportation should not be {rule}."
    return True, None


def _hard_room_type(query: Dict[str, Any], plan: Sequence[Dict[str, str]], catalog: Catalog) -> Tuple[Optional[bool], Optional[str]]:
    rule = query["local_constraint"].get("room type")
    if not rule:
        return None, None
    expected = {
        "shared room": "Shared room",
        "private room": "Private room",
        "entire room": "Entire home/apt",
    }
    for index in range(min(query["days"], len(plan))):
        stay = _field(plan[index], "accommodation")
        if not stay or stay == "-":
            continue
        name, city = get_valid_name_city(stay)
        hits = _lookup_name_city(catalog.accommodations, "NAME", "city", name, city)
        if not hits:
            continue
        room = hits[0].get("room type", "")
        if rule == "not shared room" and room == "Shared room":
            return False, f"The room type should be {rule}."
        if rule in expected and room != expected[rule]:
            return False, f"The room type should be {rule}."
    return True, None


def commonsense_checks(query: Dict[str, Any], plan: Sequence[Dict[str, str]], catalog: Catalog) -> Dict[str, Check]:
    return {
        "is_reasonable_visiting_city": is_reasonable_visiting_city(query, plan),
        "is_valid_restaurants": is_valid_restaurants(query, plan),
        "is_valid_attractions": is_valid_attractions(query, plan),
        "is_valid_accommodation": is_valid_accommodation(query, plan, catalog),
        "is_valid_transportation": is_valid_transportation(query, plan),
        "is_valid_information_in_current_city": is_valid_information_in_current_city(query, plan),
        "is_valid_information_in_sandbox": is_valid_information_in_sandbox(query, plan, catalog),
        "is_not_absent": is_not_absent(query, plan),
    }


def hard_checks(query: Dict[str, Any], plan: Sequence[Dict[str, str]], catalog: Catalog) -> Dict[str, Tuple[Optional[bool], Optional[str]]]:
    cost = get_total_cost(query, plan, catalog)
    return {
        "valid_cost": (bool(cost <= query["budget"]), None if cost <= query["budget"] else f"Cost {cost} exceeds budget {query['budget']}."),
        "valid_room_rule": _hard_room_rule(query, plan, catalog),
        "valid_cuisine": _hard_cuisine(query, plan, catalog),
        "valid_room_type": _hard_room_type(query, plan, catalog),
        "valid_transportation": _hard_transport(query, plan),
    }


def _with_city_state(record: Dict[str, Any], catalog: Catalog) -> Dict[str, Any]:
    query = _query_view(record)
    if not query["city_state"]:
        query["city_state"] = dict(catalog.city_state)
    org = query["org"]
    if org and org not in query["city_state"]:
        # Endpoints are not checked against the destination state. Keep the
        # departure city in the map so a closed route is not rejected as unknown.
        query["city_state"][org] = catalog.city_state.get(org, "")
    return query


def evaluate_plan(record: Dict[str, Any], plan: Optional[Sequence[Dict[str, str]]]) -> Dict[str, Any]:
    catalog = Catalog(record.get("reference_information"))
    query = _with_city_state(record, catalog)
    delivered = bool(plan)
    if not delivered:
        commonsense = {key: (False, "No plan delivered.") for key in COMMONSENSE_KEYS}
        hard = None
        hard_gate = False
    else:
        commonsense = commonsense_checks(query, plan, catalog)
        hard_gate = bool(commonsense["is_not_absent"][0] and commonsense["is_valid_information_in_sandbox"][0])
        hard = hard_checks(query, plan, catalog) if hard_gate else None

    commonsense_passed = sum(1 for key in COMMONSENSE_KEYS if commonsense[key][0])
    applicable = []
    if hard:
        for key, (ok, _message) in hard.items():
            if ok is None:
                continue
            applicable.append(bool(ok))
    hard_applicable = len(applicable)
    hard_passed = sum(1 for ok in applicable if ok) if hard_gate else 0
    if not hard_gate:
        # Denominator still counts constraints that apply to the query, matching
        # the official micro-average, which does not drop undelivered cases.
        constraint = query["local_constraint"]
        hard_applicable = 1  # budget always applies
        hard_applicable += int(bool(constraint.get("house rule")))
        hard_applicable += int(bool(constraint.get("cuisine")))
        hard_applicable += int(bool(constraint.get("room type")))
        hard_applicable += int(bool(constraint.get("transportation")))
        hard_passed = 0

    final_pass = delivered and commonsense_passed == len(COMMONSENSE_KEYS) and hard_gate and hard_passed == hard_applicable
    return {
        "delivered": delivered,
        "final_pass": final_pass,
        "commonsense": commonsense,
        "hard": hard,
        "commonsense_passed": commonsense_passed if delivered else 0,
        "commonsense_total": len(COMMONSENSE_KEYS),
        "hard_passed": hard_passed,
        "hard_applicable": hard_applicable,
    }


def evaluate_text(record: Dict[str, Any], text: str) -> Dict[str, Any]:
    return evaluate_plan(record, parse_plan(text))


def plan_is_correct(text: str, record: Dict[str, Any]) -> bool:
    try:
        return bool(evaluate_text(record, text)["final_pass"])
    except Exception:
        return False
