#!/usr/bin/env python3
"""
User query generator for Japan-wide multi-day travel plans.

Updated to:
- RANDOMIZE inputs and ensure a mix of ROUND-TRIP vs OPEN-JAW.
- Explicitly mention total_days inside generated_query (e.g., "5-day trip").
- Add start_city, end_city, and desired_cities.
- Embed all subfields (time_of_day, duration, seasonality, budget, companion,
  key activities & atmospheres) inside generated_query.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


# =========================
# Config
# =========================

@dataclass
class Config:
    poi_db_path: Path
    output_path: Path
    model: str = "gpt-5"
    num_queries: int = 30


# =========================
# Data loading
# =========================

def load_poi_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        print(f"[ERROR] Database file not found at: {path}")
        return []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                print("[WARN] Skipping invalid JSON line.")
    return records


# =========================
# Aggregation (Randomized)
# =========================

def aggregate_random_profile(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collects unique possibilities and picks a random sample to ensure variety.
    Also aggregates cities so the LLM can choose start/end + desired cities.
    """
    landscape_pool = set()
    activity_pool = set()
    atmosphere_pool = set()

    time_of_day_pool = set()
    duration_pool = set()
    seasonality_pool = set()
    city_theme_pool = set()
    city_pool = set()

    for rec in records:
        if rec.get("landscape_contents"):
            landscape_pool.update(rec["landscape_contents"])
        if rec.get("activities"):
            activity_pool.update(rec["activities"])
        if rec.get("atmospheres"):
            atmosphere_pool.update(rec["atmospheres"])

        if rec.get("time_of_day_counts"):
            time_of_day_pool.update(rec["time_of_day_counts"].keys())
        if rec.get("duration_counts"):
            duration_pool.update(rec["duration_counts"].keys())
        if rec.get("seasonality_counts"):
            seasonality_pool.update(rec["seasonality_counts"].keys())

        theme = rec.get("city_theme")
        if isinstance(theme, str):
            city_theme_pool.add(theme)
        elif isinstance(theme, list):
            city_theme_pool.update(theme)
        if rec.get("city_theme_counts"):
            city_theme_pool.update(rec["city_theme_counts"].keys())

        # Cities (both raw and normalized if available)
        city = rec.get("city")
        if isinstance(city, str) and city.strip():
            city_pool.add(city.strip())

        city_norm = rec.get("city_normalized")
        if isinstance(city_norm, str) and city_norm.strip():
            city_pool.add(city_norm.strip())

    def pick_random(pool_set, n):
        items = list(pool_set)
        if not items:
            return []
        if len(items) <= n:
            random.shuffle(items)
            return items
        return random.sample(items, n)

    return {
        "country": "Japan",
        "num_pois_scanned": len(records),
        "random_landscape_phrases": pick_random(landscape_pool, 30),
        "random_activity_phrases": pick_random(activity_pool, 30),
        "random_atmosphere_phrases": pick_random(atmosphere_pool, 30),
        "random_time_of_day_options": pick_random(time_of_day_pool, 6),
        "random_duration_options": pick_random(duration_pool, 6),
        "random_seasonality_options": pick_random(seasonality_pool, 8),
        "random_city_theme_options": pick_random(city_theme_pool, 15),
        "random_city_options": pick_random(city_pool, 20),
    }


# =========================
# Prompt
# =========================

def build_llm_prompt(profile: Dict[str, Any], num_queries: int) -> str:
    profile_str = json.dumps(profile, ensure_ascii=False, indent=2)

    # Examples now include start_city, end_city, desired_cities,
    # and explicitly mention total_days and cities in generated_query.
    examples = """
Here are example JSONL lines. Notice the difference between round-trip and open-jaw,
and how total_days & cities are mentioned inside generated_query:

{"qid": "Q1",
 "country": "Japan",
 "total_days": 7,
 "time_of_day_pref": ["morning", "evening"],
 "duration_pref": ["multi_day"],
 "seasonality_pref": ["spring"],
 "travel_companion_category": "partner",
 "round_trip": true,
 "budget_style": "midrange",
 "start_city": "Tokyo",
 "end_city": "Tokyo",
 "desired_cities": ["Tokyo", "Kyoto", "Nara"],
 "atmosphere_pref": ["peaceful"],
 "activity_pref": ["walking"],
 "generated_query": "I'm planning a 7-day round-trip in Japan with my partner, flying into and out of Tokyo. We'd love mostly peaceful morning walks and quiet evening strolls, especially around Kyoto and maybe a side trip to Nara. We're on a midrange budget and want a reflective spring trip when cherry blossoms are out, without overpacking the schedule."}

{"qid": "Q2",
 "country": "Japan",
 "total_days": 10,
 "time_of_day_pref": ["night"],
 "duration_pref": ["multi_day"],
 "seasonality_pref": ["winter"],
 "travel_companion_category": "alone",
 "round_trip": false,
 "budget_style": "economic",
 "start_city": "Sapporo",
 "end_city": "Tokyo",
 "desired_cities": ["Sapporo", "Hakodate", "Tokyo"],
 "atmosphere_pref": ["lively"],
 "activity_pref": ["drinking", "food"],
 "generated_query": "I'm a solo traveler planning a 10-day open-jaw winter trip in Japan, flying into Sapporo and flying out of Tokyo so I don't have to backtrack. I want lively neighborhoods with great food and casual bars at night, maybe stopping in Hakodate before ending in Tokyo. I'm on a tight budget, happy with small guesthouses and hostels, and I'm mainly interested in nightlife and local food rather than daytime sightseeing."}
""".strip()

    return f"""
You are designing diverse user search queries for a Japan-wide travel system.

Below is a **random selection** of attributes found in our database:
{profile_str}

Here are example formats and styles:
{examples}

Your task: Generate **{num_queries} JSONL lines**.

====================================================
CRITICAL INSTRUCTION ON TRIP TYPE & CITIES:
====================================================
- You MUST generate a mix of trip types.
- Approximately **50%** should be `"round_trip": true` (Start/End same city).
- Approximately **50%** should be `"round_trip": false` (Open-jaw: Start/End different cities).
- For each line, you MUST set:
    - "start_city": a city in Japan.
    - "end_city": a city in Japan (same as start_city if round_trip = true; different if false).
    - "desired_cities": a list of 2–5 cities (including start_city and end_city) that the traveler hopes to visit.
- The `generated_query` text MUST:
    - Explicitly mention the **trip length** using `total_days` (e.g., "a 5-day trip" or "for 10 days").
    - Explicitly mention the **start and end cities** (e.g., "flying into Osaka and out of Tokyo" or "round-trip in and out of Tokyo").
    - Mention some or all of the cities in "desired_cities" as places they want to visit.
    - NOT fully arrange a day-by-day schedule. It should just say they want to visit those cities during the trip.

====================================================
SUBFIELDS TO EMBED IN THE NATURAL QUERY:
====================================================
For EACH JSON line:
- "total_days": an integer between 2 and 21.
- "time_of_day_pref": 1–3 values chosen from random_time_of_day_options.
- "duration_pref": 1–3 values chosen from random_duration_options.
- "seasonality_pref": 1 value (or small list) chosen from random_seasonality_options.
- "travel_companion_category": choose a persona like "alone", "partner", "family", "friends", etc.
- "round_trip": boolean (true/false) as described above.
- "budget_style": one of ["economic", "midrange", "luxury"] or similar.
- "atmosphere_pref": 1–3 phrases chosen from random_atmosphere_phrases.
- "activity_pref": 2–4 phrases chosen from random_activity_phrases.

In the **generated_query** (2–5 sentences), you MUST weave in:
- The total number of days.
- The start and end cities and the open-jaw vs round-trip nature.
- At least one seasonality phrase (e.g. "during cherry blossom season", "in winter", etc.).
- The budget style (e.g. "mid-range budget", "tight budget", "splurge/luxury").
- The travel companion (e.g. "solo traveler", "with my partner", "with friends", "with family").
- 2–3 concrete activity phrases (e.g. "sightseeing and photography", "bar-hopping and street food", etc.).
- 1–2 atmosphere phrases (e.g. "serene", "lively and energetic", "historic and contemplative").

====================================================
OUTPUT FORMAT:
====================================================
Required JSON keys for each line:
- "qid"
- "country"
- "total_days"
- "time_of_day_pref"
- "duration_pref"
- "seasonality_pref"
- "travel_companion_category"
- "round_trip"
- "budget_style"
- "start_city"
- "end_city"
- "desired_cities"
- "atmosphere_pref"
- "activity_pref"
- "generated_query"

- "country" MUST be "Japan".
- IMPORTANT: Output ONLY raw JSON objects, **one per line** (standard JSONL),
  with no commentary, no markdown, and no trailing commas.
""".strip()


# =========================
# LLM call
# =========================

def call_llm(client: OpenAI, cfg: Config, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt = build_llm_prompt(profile, cfg.num_queries)

    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {
                "role": "system",
                "content": "You output ONLY JSONL lines: one valid JSON object per line, no extra commentary."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    raw = resp.choices[0].message.content or ""
    results: List[Dict[str, Any]] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("```"):
            # In case the model wraps output in a code block, just skip fences.
            continue
        try:
            results.append(json.loads(line))
        except Exception as e:
            print("[WARN] Bad JSON line, skipping:", e, "LINE:", line[:200])
    return results


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poi-db", type=Path, default=Path("data/poi_database.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/user_queries_japan.jsonl"))
    parser.add_argument("--model", type=str, default="gpt-5")
    parser.add_argument("--num-queries", type=int, default=20)
    args = parser.parse_args()

    cfg = Config(
        poi_db_path=args.poi_db,
        output_path=args.output,
        model=args.model,
        num_queries=args.num_queries,
    )

    print("[INFO] Loading POI DB ...")
    records = load_poi_records(cfg.poi_db_path)

    if not records:
        print("[ERROR] No records loaded. Exiting.")
        return

    # Initial shuffle
    random.shuffle(records)

    print("[INFO] Building randomized profile ...")
    profile = aggregate_random_profile(records)

    print("[INFO] Calling LLM ...")
    client = OpenAI()
    raw_queries = call_llm(client, cfg, profile)
    queries = raw_queries[: cfg.num_queries]

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Writing output ...")
    with cfg.output_path.open("w", encoding="utf-8") as f:
        for i, obj in enumerate(queries, start=1):
            # Force our own qid format:
            obj["qid"] = f"q{i:05d}"

            obj.setdefault("country", "Japan")
            obj.setdefault("travel_companion_category", "alone")
            obj.setdefault("budget_style", "midrange")
            obj.setdefault("round_trip", random.choice([True, False]))
            obj.setdefault("total_days", random.randint(2, 21))

            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


    print(f"[DONE] Saved {len(queries)} queries → {cfg.output_path}")


if __name__ == "__main__":
    main()
