#!/usr/bin/env python3
"""
Agent-based multi-day itinerary planner from pre-selected top-k POIs.

- Inputs:
    * data/poi_cards_structured.json        (full POI DB with lat/lon)
    * data/user_queries_japan.jsonl         (contains qid, total_days, prefs)
    * topk_pick/topk_{qid}.json                  (top-k POIs for that query, from matcher)

- Process:
    * Enrich top-k POIs with card-level data (lat/lon, etc.).
    * Extract unique cities and compute city centroids.
    * For each ordered city pair, query Google Maps Directions API
      to get realistic travel info (using driving as schedule-independent proxy).
    * Package:
        - query metadata
        - candidate POIs
        - cities + city scores
        - city_routes (origin, dest, route_info)
        - intra_city_distance_hints (approx km between POIs in same city)
      and send to an LLM "planning agent" which returns a JSON itinerary.

- Output:
    * itineraty/itinerary_agent_{qid}.json

Environment variables:
    - OPENAI_API_KEY (for OpenAI LLM)
    - GOOGLE_MAPS_API_KEY (for Google Maps Directions)

Requires:
    - pip install openai requests
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import requests
from openai import OpenAI


# ========================
# Config
# ========================

@dataclass
class Config:
    poi_cards_path: Path = Path("data/poi_cards_clean_structured.json")
    user_queries_path: Path = Path("data/user_queries_japan.jsonl")

    topk_dir: Path = Path(".")
    output_dir: Path = Path(".")

    # LLM
    llm_model: str = "gpt-5"

    # Google Maps
    gmaps_api_key: Optional[str] = None
    gmaps_mode: str = "transit"  # unused now, but kept for possible future tweak
    gmaps_cache_path: Path = Path("_cache/gmaps_routes_cache.json")

    # Planning heuristics for prompt shaping
    max_pois_for_prompt: int = 40  # cap how many candidate POIs we show the agent


client = OpenAI()


# ========================
# Data loading
# ========================

def load_poi_cards(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


def load_queries_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qid")
            if qid is not None:
                out[qid] = obj
    return out


def load_topk_results(topk_dir: Path, qid: str) -> List[Dict[str, Any]]:
    """
    Expects topk_pick/topk_{qid}.json to be a list of POI candidate dicts like:

    [
      {
        "poi_name": "...",
        "city": "...",
        "score": 0.85,
        "types": [...],
        "landscape": [...],
        "activities": [...],
        "atmosphere": [...],
        "season_primary": [...],
        "season_secondary": [...]
      },
      ...
    ]
    """
    topk_path = topk_dir / f"topk_{qid}.json"
    if not topk_path.exists():
        raise SystemExit(f"[ERROR] top-k file not found: {topk_path}")
    with topk_path.open() as f:
        return json.load(f)


# ========================
# POI joining / indexing
# ========================

def build_poi_index(poi_cards: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Index full POI cards by (poi_name, city), key-lowered strings.
    """
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for card in poi_cards:
        name = (card.get("poi_name") or "").strip()
        city = (card.get("city") or "").strip()
        if not name or not city:
            continue
        key = (name.lower(), city.lower())
        if key not in index:
            index[key] = card
    return index


def attach_card_data_to_topk(
    topk: List[Dict[str, Any]],
    poi_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Enrich top-k entries with lat/lon and any card-level fields from data/poi_cards_structured.json.
    Assumes each top-k item has "poi_name", "city", "score", etc.
    """
    enriched: List[Dict[str, Any]] = []
    for item in topk:
        name = (item.get("poi_name") or "").strip()
        city = (item.get("city") or "").strip()
        key = (name.lower(), city.lower())
        card = poi_index.get(key, {})

        merged = dict(item)
        # attach coordinates
        for field in ("lat", "lon"):
            if field in card:
                merged[field] = card[field]

        # optionally attach other fields you care about
        for field in ("address", "url", "tags"):
            if field in card and field not in merged:
                merged[field] = card[field]

        enriched.append(merged)
    return enriched


# ========================
# City grouping & centroids
# ========================

def group_pois_by_city(pois: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    city_to_pois: Dict[str, List[Dict[str, Any]]] = {}
    for p in pois:
        city = (p.get("city") or "").strip()
        if not city:
            continue
        city_to_pois.setdefault(city, []).append(p)

    # sort by score desc within each city
    for city, lst in city_to_pois.items():
        lst.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return city_to_pois


def compute_city_scores(city_to_pois: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for city, plist in city_to_pois.items():
        scores[city] = sum(p.get("score", 0.0) for p in plist)
    return scores


def compute_city_centroids_from_pois(
    poi_cards: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Approximate city centroid as average of lat/lon across all POIs in that city.
    Currently not used in GM requests (we just use city names), but kept for
    future refinement.
    """
    accum: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for card in poi_cards:
        city = (card.get("city") or "").strip()
        lat = card.get("lat")
        lon = card.get("lon")
        if not city or lat is None or lon is None:
            continue

        if city not in accum:
            accum[city] = {"lat_sum": 0.0, "lon_sum": 0.0}
            counts[city] = 0
        accum[city]["lat_sum"] += float(lat)
        accum[city]["lon_sum"] += float(lon)
        counts[city] += 1

    centroids: Dict[str, Dict[str, float]] = {}
    for city, a in accum.items():
        c = counts[city]
        centroids[city] = {
            "lat": a["lat_sum"] / c,
            "lon": a["lon_sum"] / c,
        }
    return centroids


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Approximate great-circle distance between two points on Earth (in km).
    """
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def build_intra_city_distance_hints(
    candidate_pois: List[Dict[str, Any]],
    max_distance_km: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Build approximate intra-city distance hints between candidate POIs.

    Returns a list of dicts like:
      {
        "city": "Tokyo",
        "from_poi_name": "Tokyo Skytree",
        "to_poi_name": "Senso-ji",
        "approx_distance_km": 3.2
      }

    We include BOTH directions (A->B and B->A) so the LLM can match either way.
    """
    # group by city
    by_city: Dict[str, List[Dict[str, Any]]] = {}
    for p in candidate_pois:
        city = (p.get("city") or "").strip()
        if not city:
            continue
        lat = p.get("lat")
        lon = p.get("lon")
        name = (p.get("poi_name") or "").strip()
        if not name or lat is None or lon is None:
            continue
        by_city.setdefault(city, []).append(p)

    hints: List[Dict[str, Any]] = []

    for city, plist in by_city.items():
        n = len(plist)
        for i in range(n):
            p1 = plist[i]
            name1 = (p1.get("poi_name") or "").strip()
            lat1 = p1.get("lat")
            lon1 = p1.get("lon")
            if name1 == "" or lat1 is None or lon1 is None:
                continue

            for j in range(i + 1, n):
                p2 = plist[j]
                name2 = (p2.get("poi_name") or "").strip()
                lat2 = p2.get("lat")
                lon2 = p2.get("lon")
                if name2 == "" or lat2 is None or lon2 is None:
                    continue

                d = haversine_km(float(lat1), float(lon1), float(lat2), float(lon2))
                if d > max_distance_km:
                    # extremely unlikely to be an intra-city move
                    continue

                d_rounded = round(d, 1)

                # A -> B
                hints.append(
                    {
                        "city": city,
                        "from_poi_name": name1,
                        "to_poi_name": name2,
                        "approx_distance_km": d_rounded,
                    }
                )
                # B -> A
                hints.append(
                    {
                        "city": city,
                        "from_poi_name": name2,
                        "to_poi_name": name1,
                        "approx_distance_km": d_rounded,
                    }
                )

    return hints


# =====================
# Google Maps integration (always use GM when possible)
# =====================

def _format_city_for_gmaps(city: str) -> str:
    """
    Turn a city label like 'Osaka' into something gmaps can reliably route to.
    Very simple heuristic: '<city>, Japan'.
    """
    if not city:
        return ""
    city = city.strip()
    if "japan" in city.lower():
        return city
    return f"{city}, Japan"


def _gmaps_directions_request(origin: str, destination: str, mode: str) -> Optional[Dict[str, Any]]:
    """
    Low-level wrapper for a single Directions API call.
    Returns None on any error or non-OK status.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return None

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "region": "jp",
        "language": "en",
        "key": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
    except Exception as e:
        print(f"[WARN] Directions request failed for {origin} -> {destination} ({mode}): {e}")
        return None

    if resp.status_code != 200:
        print(f"[WARN] Directions HTTP {resp.status_code} for {origin} -> {destination} ({mode})")
        return None

    data = resp.json()
    status = data.get("status", "UNKNOWN")

    # Only one warning line per failed call
    if status != "OK" or not data.get("routes"):
        print(f"[WARN] Directions status={status} for {origin} -> {destination} ({mode})")
        return None

    route = data["routes"][0]
    leg = route["legs"][0]
    return {
        "distance_meters": leg["distance"]["value"],
        "duration_seconds": leg["duration"]["value"],
        "start_address": leg.get("start_address"),
        "end_address": leg.get("end_address"),
        "overview_polyline": route.get("overview_polyline", {}).get("points"),
    }


def fetch_gmaps_leg(origin_city: str, destination_city: str) -> Optional[Dict[str, Any]]:
    """
    High-level routing helper:

      - We ALWAYS try to get *some* route from Google Maps.
      - Prefer 'driving' because it does not depend on transit schedules,
        so it works even at midnight.
      - If driving somehow fails, we *then* try 'transit'.
      - If both fail, return None and the caller falls back to heuristics.
    """
    origin = _format_city_for_gmaps(origin_city)
    destination = _format_city_for_gmaps(destination_city)

    # 1) Driving first (stable, schedule independent)
    data = _gmaps_directions_request(origin, destination, mode="driving")
    if data is not None:
        return data

    # 2) Transit as a backup (if available)
    data = _gmaps_directions_request(origin, destination, mode="transit")
    if data is not None:
        return data

    # 3) Nothing worked: caller will use heuristics
    return None


def recommend_mode(
    budget_style: str,
    distance_km: Optional[float],
    context: str = "inter_city",
) -> str:
    """
    Heuristic mode recommendation based on budget and distance.

    Parameters
    ----------
    budget_style : str
        e.g. "luxury", "midrange", "mid-range", "budget", "economic".
    distance_km : Optional[float]
        Driving distance in kilometers (from Google Maps for inter-city,
        or approximate haversine/GM for intra-city). Can be None.
    context : str
        "inter_city"  -> city-to-city transfers (use Shinkansen/bus/JR)
        "intra_city"  -> within a single city (walk/metro/JR/bus/taxi)

    Returns
    -------
    str : human-readable transport recommendation, e.g.
        "walk", "metro / JR", "taxi", "Shinkansen (reserved seat)", etc.
    """
    style = (budget_style or "").lower()
    if style in {"economy", "economic"}:
        style = "budget"
    if style in {"mid-range", "mid range"}:
        style = "midrange"

    # -------- Intra-city logic (within one city) --------
    if context == "intra_city":
        # No distance: fall back to a generic mode per budget tier
        if distance_km is None:
            if style == "luxury":
                return "taxi"
            if style == "budget":
                return "walk / metro / JR"
            return "walk / metro / JR / bus"

        # Very short hops
        if distance_km <= 1.5:
            if style == "luxury":
                return "walk or short taxi ride"
            if style == "budget":
                return "walk"
            return "walk (or short metro/JR if convenient)"

        # Medium within-city distances
        if distance_km <= 5.0:
            if style == "luxury":
                return "taxi"
            if style == "budget":
                return "metro / JR / local bus"
            return "metro / JR (taxi if tired or late)"

        # Longer cross-town moves
        if style == "luxury":
            return "taxi"
        if style == "budget":
            return "metro / JR / local bus"
        return "JR / metro (taxi only if needed)"

    # -------- Inter-city logic (between two cities) --------
    # If we have no distance at all, fall back to generic labels
    if distance_km is None:
        if style == "luxury":
            return "Shinkansen / private transfer"
        if style == "budget":
            return "standard Shinkansen / highway bus"
        return "JR limited express / standard Shinkansen"

    # Short hops (Kansai-style: Osaka–Kyoto–Nara–Kobe, etc.)
    if distance_km < 60:
        if style == "luxury":
            return "JR limited express / taxi to station"
        if style == "budget":
            return "local JR / rapid train"
        return "local JR / rapid train"

    # Medium distances (e.g. Osaka–Hiroshima, Nagoya–Kanazawa)
    if distance_km < 250:
        if style == "luxury":
            return "Shinkansen Green Car"
        if style == "budget":
            return "standard Shinkansen or highway bus"
        return "Shinkansen (reserved seat)"

    # Long distances (Tokyo–Hiroshima, Tokyo–Fukuoka, etc.)
    if style == "luxury":
        return "Shinkansen Green Car (or domestic flight)"
    if style == "budget":
        return "standard Shinkansen / overnight highway bus"
    return "Shinkansen (reserved seat)"


def build_city_routes_for_agent(
    cfg: Config,
    cities: List[str],
    centroids: Dict[str, Dict[str, float]],
    budget_style: str,
) -> List[Dict[str, Any]]:
    """
    Build a list of routes between ordered city pairs for the agent.

    Each route dict:

    {
      "origin_city": "...",
      "dest_city": "...",
      "mode": "<recommended transit mode>",
      "duration_minutes": <int or null>,
      "distance_km": <float or null>,
      "summary": "<string or null>",
      "gmaps_directions_link": "<google maps directions URL>"
    }
    """
    routes: List[Dict[str, Any]] = []
    budget_style = budget_style or "midrange"

    for i, origin in enumerate(cities):
        for j, dest in enumerate(cities):
            if origin == dest:
                continue

            print(f"[INFO] GM route: {origin} -> {dest}")
            g = fetch_gmaps_leg(origin, dest)

            distance_km: Optional[float] = None
            duration_minutes: Optional[int] = None
            summary: Optional[str] = None

            if g is not None:
                distance_km = g["distance_meters"] / 1000.0
                duration_minutes = g["duration_seconds"] // 60
                summary = f"{round(distance_km, 1)} km, ~{int(duration_minutes)} min (GM driving proxy)"
            else:
                print(f"[WARN] No GM route found for {origin} -> {dest}; using heuristics.")

            mode = recommend_mode(budget_style, distance_km)

            # Simple GM directions link (we still show transit as travelmode)
            gmaps_link = (
                "https://www.google.com/maps/dir/?api=1"
                f"&origin={origin.replace(' ', '+')}+Japan"
                f"&destination={dest.replace(' ', '+')}+Japan"
                "&travelmode=transit"
            )

            routes.append(
                {
                    "origin_city": origin,
                    "dest_city": dest,
                    "mode": mode,
                    "duration_minutes": duration_minutes,
                    "distance_km": distance_km,
                    "summary": summary,
                    "gmaps_directions_link": gmaps_link,
                }
            )

    return routes


# ========================
# Agent prompt & call
# ========================

PLANNER_SYSTEM_PROMPT = """
You are an expert Japanese travel planner and routing agent.

You will receive:
- A user query object (with total_days, season, time-of-day preferences, budget_style, etc.).
- A list of CANDIDATE POIs (with city, score, lat/lon, and attributes).
- A list of CITIES derived from those POIs, with city_scores.
- A list of CITY ROUTES between pairs of cities, each including:
  { "origin_city", "dest_city", "mode", "duration_minutes", "distance_km",
    "summary", "gmaps_directions_link" }
- A list of INTRA-CITY DISTANCE HINTS:
  [
    {
      "city": "<city>",
      "from_poi_name": "<poi>",
      "to_poi_name": "<poi>",
      "approx_distance_km": <float>
    },
    ...
  ]
  giving approximate distances between candidate POIs in the same city.

Your tasks:

1. Design a realistic MULTI-DAY itinerary that:
   - Uses the given "total_days" exactly (if provided).
   - Picks which cities to visit and in what order.
   - Schedules intra-city days with 2–4 POI visits per full day.
   - Uses shorter travel days (1–2 POIs) when a long inter-city transfer happens.
   - Respects the user's season, atmosphere, and activity preferences as much as possible.
   - Respects the user's budget_style when choosing transport modes:
     * "luxury": more taxi/private transfers, Shinkansen Green Car.
     * "midrange" / "mid-range": mix of metro/JR and occasional taxi; standard Shinkansen.
     * "budget"/"economic": mostly walk + metro/JR/bus, avoid taxis where possible.

2. For CITY-TO-CITY transitions:
   - WHENEVER the itinerary moves from one base city to another, you MUST insert
     a segment of type "city_transfer".
   - For that segment, choose the appropriate city route from the provided CITY ROUTES
     where origin_city and dest_city match.
   - Copy the route's "mode", "duration_minutes", "distance_km", "summary", and
     "gmaps_directions_link" into the "gmaps_route" object for that segment.
   - Never invent your own inter-city durations: always re-use the given route values.
   - You may rephrase the mode in a more detailed way, but:
       * The JSON field gmaps_route.mode should remain semantically consistent with
         the recommended mode (e.g., Shinkansen vs highway bus vs JR rapid train).

   - For EACH city_transfer segment you MUST ALSO output:
     {
       "segment_type": "city_transfer",
       "from_city": "<string>",
       "to_city": "<string>",
       "detailed_mode": "<string>",
       "estimated_cost_jpy": <int>,
       "budget_comment": "<short sentence>",
       "gmaps_route": {
         "origin_city": "<string>",
         "dest_city": "<string>",
         "mode": "<string>",
         "duration_minutes": <int or null>,
         "distance_km": <float or null>,
         "summary": "<string or null>",
         "gmaps_directions_link": "<string>"
       }
     }

     * detailed_mode:
       - A concrete, specific description of the likely inter-city transport, e.g.:
         - "JR Tokaido Shinkansen (Nozomi) reserved seat"
         - "JR Sanyo Shinkansen (Sakura/Hikari) reserved seat"
         - "JR Kyoto Line Special Rapid Service"
         - "JR Kagoshima Main Line limited express"
         - "Overnight highway bus between Fukuoka and Hiroshima"
       - Avoid vague phrases like "local JR / rapid train" alone; always specify a line
         or service type when possible, consistent with origin and destination cities.
       - It does NOT need to match real-time timetables; it must be plausible and
         consistent with gmaps_route.mode and distance_km.

     * estimated_cost_jpy:
       - A single integer representing a rough per-person fare in Japanese yen for
         the full city-to-city leg.
       - Use the budget_style, mode and distance_km to choose a reasonable range:
         • If mode mentions "Shinkansen":
             - For distance_km < 150 km:
                 -> choose a cost in the 4_000–8_000 JPY range.
             - For 150 km <= distance_km <= 300 km:
                 -> choose a cost in the 8_000–15_000 JPY range.
             - For distance_km > 300 km:
                 -> choose a cost in the 15_000–25_000 JPY range.
         • If mode mentions "JR limited express" or "local JR / rapid train":
             - For distance_km < 100 km:
                 -> choose a cost in the 2_000–5_000 JPY range.
             - For 100 km <= distance_km <= 250 km:
                 -> choose a cost in the 5_000–10_000 JPY range.
         • If mode mentions "highway bus" or "overnight highway bus":
             - For medium–long distances, choose a cost in the 4_000–10_000 JPY range,
               often cheaper than Shinkansen for the same distance.
         • If mode mentions "domestic flight":
             - Choose a cost in the 10_000–25_000 JPY range.

       - Modify your choice slightly based on budget_style:
         • LUXURY:
             - Prefer the upper half of the appropriate range (e.g. Green Car, reserved seats).
         • MIDRANGE:
             - Use the middle of the range.
         • BUDGET / ECONOMIC:
             - Prefer the lower half of the range (e.g. non-reserved Shinkansen seats,
               highway bus instead of Shinkansen if that matches the mode string).
       - If distance_km is null, infer a reasonable bucket from the city pair
         (short: Kansai-type hops, medium: regional, long: cross-country) and mode.

     * budget_comment:
       - 1 short sentence tying together mode, time and cost, e.g.:
         - "Standard reserved-seat Shinkansen tickets cost around 9,000 JPY for this midrange hop."
         - "A highway bus keeps costs lower (about 5,000 JPY) but takes longer than the train."
         - "Green Car seats are more comfortable for this long ride at roughly 20,000 JPY."

3. For WITHIN-DAY, WITHIN-CITY transitions:
   - BETWEEN each pair of visit segments on the SAME day (same base_city),
     you MUST insert an "intra_city_transfer" segment.
   - The sequence on a day should look like:
       visit -> intra_city_transfer -> visit -> intra_city_transfer -> visit ...
     (Start/end of day may have a visit with no preceding/following transfer.)

   - You are given "intra_city_distance_hints", which contains objects:
     { "city", "from_poi_name", "to_poi_name", "approx_distance_km" }.
     * When you create an intra_city_transfer between two visits:
       - Look up a hint where:
           city == that day’s base_city AND
           from_poi_name == the previous visit's poi_name AND
           to_poi_name   == the next visit's poi_name.
       - If a matching hint exists, treat approx_distance_km as the rough distance
         for that transfer.
       - Use this approximate distance to:
         • Choose recommended_mode (short distance → walk; medium → metro/JR; long → metro/JR or taxi depending on budget_style).
         • Choose estimated_duration_minutes (shorter times for short distances, longer for long distances).
         • If the recommended_mode includes taxi, choose the fare bucket based on approx_distance_km.
     * If NO hint exists for a pair of POIs, fall back to your own distance intuition
       based on the city and POI descriptions, and still respect the budget_style rules
       and time-of-day constraints.

   - For each "intra_city_transfer" you MUST output:

     {
       "segment_type": "intra_city_transfer",
       "from_city": "<string>",
       "to_city": "<string>",
       "from_poi_name": "<string>",
       "to_poi_name": "<string>",
       "recommended_mode": "<string>",
       "transition_name": "<string>",
       "estimated_duration_minutes": <int>,
       "estimated_cost_jpy": <int>,
       "budget_comment": "<short sentence>"
     }

   - Set fields as follows:

     * segment_type:
       - Always "intra_city_transfer".

     * from_city / to_city:
       - Cities of the previous and next visit segments (usually identical).

     * from_poi_name / to_poi_name:
       - Exactly the "poi_name" values of the previous and next visit segments.

     * recommended_mode:
       - One of:
         - "walk"
         - "metro / subway"
         - "JR / local rail"
         - "local bus"
         - "taxi"
         - "walk + taxi"
         - "walk + metro / JR"
       - Choose based on budget_style and distance:
         - LUXURY:
           • short distance in compact area: "walk" or "walk + taxi".
           • most cross-town moves: "taxi".
         - MIDRANGE:
           • short distance: "walk".
           • cross-town: "metro / subway" or "JR / local rail".
           • "taxi" only when clearly convenient (late-night, with luggage, or to save time).
         - BUDGET / ECONOMIC:
           • short distance: "walk".
           • longer: "metro / subway", "JR / local rail", or "local bus".
           • Avoid pure "taxi" unless absolutely necessary.

     * transition_name:
       - A short, specific label for the transfer, e.g.:
         - "Walk through Asakusa shopping streets"
         - "Tokyo Metro Ginza Line ride"
         - "JR Yamanote Line from Shinjuku to Shibuya"
         - "Short taxi ride across central Kyoto"
       - If you are not sure about the exact line name, use a plausible generic name
         like "metro ride on local subway line" or "JR local train between districts".

     * estimated_duration_minutes:
       - A realistic travel time in minutes (integer).
       - Use simple heuristics, consistent with approx_distance_km when available:
         - Very short walk (<= ~1 km): 5–10 minutes.
         - Longer walk but same neighborhood (~1–2 km): 10–20 minutes.
         - Metro/JR within a compact city area (~3–8 km): 10–25 minutes including walking.
         - Cross-town metro/JR in big cities (~8–15 km): 20–40 minutes.
         - Taxi for short hops (<= ~2 km): 5–15 minutes.
         - Taxi for cross-town (~2–10+ km): 15–30 minutes.
       - Be consistent with the recommended_mode and the day structure (do not exceed
         what fits into the day’s time slots).

     * estimated_cost_jpy:
       - This is the TOTAL estimated per-person cost in Japanese yen for that transfer,
         regardless of mode (walk, metro, JR, bus, taxi, etc.).
       - If the recommended_mode is:
         • "walk":
             -> cost is 0.
         • "metro / subway" or "JR / local rail":
             -> For short hops within a city, choose a cost in the 200–400 JPY range.
             -> For longer cross-town or multi-line rides, choose a cost in the 300–800 JPY range.
         • "local bus":
             -> Choose a cost in the 200–500 JPY range, depending on distance.
         • "walk + metro / JR":
             -> Cost is that of the metro/JR portion (typically 200–500 JPY).
         • "taxi" or "walk + taxi":
             -> If approx_distance_km is available (from intra_city_distance_hints),
                use the following buckets:
                · approx_distance_km <= 2.0:
                    -> choose a fare between 700 and 1_500 JPY.
                · 2.0 < approx_distance_km <= 5.0:
                    -> choose a fare between 1_500 and 3_000 JPY.
                · approx_distance_km > 5.0:
                    -> choose a fare between 3_000 and 5_000 JPY.
               If approx_distance_km is NOT available, infer a reasonable bucket from
               the city and described distance and still choose a single integer in
               one of these ranges.
       - Always output a single integer (e.g., 0, 260, 420, 900, 1800, 3200).
       - These are rough estimates, not exact rates.

     * budget_comment:
       - 1 short sentence tying together mode, time, distance and budget_style, e.g.:
         - "Walking between these nearby temples keeps costs at zero for a budget trip."
         - "Midrange travelers can use the metro to cross town in about 20 minutes for a few hundred yen."
         - "A short taxi ride (~2,000 JPY) keeps things comfortable on a luxury itinerary."
         - "Taking the JR local line is cheap and efficient for this longer intra-city hop."

4. POI usage:
   - Only use POIs from the CANDIDATE list.
   - Prefer higher-score POIs, but also consider diversity of landscape/activities/atmosphere.
   - For each visit segment, you MUST:
     - Use the "poi_name" and "city" from a candidate POI.
     - Copy that POI's "landscape", "activities" and "atmosphere" arrays into the segment.
       (You may subset or reorder them for brevity, but do NOT invent entirely new items.)
     - Choose a "duration_label" for the visit:
       * One of: ["short", "half_day", "full_day", "multi_day"].
       * If the candidate POI has "duration_counts", base the label on the most
         common duration; otherwise infer a reasonable label from the description.
   - For each visit segment, specify:
     - "segment_type": "visit".
     - "time_slot": one of ["morning", "afternoon", "evening"].
     - "poi_name", "city".
     - "landscape": array of strings.
     - "activities": array of strings.
     - "atmosphere": array of strings.
     - "duration_label": "short" | "half_day" | "full_day" | "multi_day".
   - You may skip some candidate POIs if they don't fit the time window.

5. Output format:
   - You MUST return a single valid JSON object with this shape:

   {
     "qid": "<qid>",
     "summary": "<1-3 sentence natural-language summary>",
     "days": [
       {
         "day_index": <1-based integer>,
         "base_city": "<city for the night>",
         "day_theme": "<short descriptive label>",
         "segments": [
           {
             "segment_type": "visit",
             "time_slot": "morning" | "afternoon" | "evening",
             "poi_name": "<string>",
             "city": "<string>",
             "landscape": [ "<string>", ... ],
             "activities": [ "<string>", ... ],
             "atmosphere": [ "<string>", ... ],
             "duration_label": "short" | "half_day" | "full_day" | "multi_day"
           },
           {
             "segment_type": "intra_city_transfer",
             "from_city": "<string>",
             "to_city": "<string>",
             "from_poi_name": "<string>",
             "to_poi_name": "<string>",
             "recommended_mode": "<string>",
             "transition_name": "<string>",
             "estimated_duration_minutes": <int>,
             "estimated_cost_jpy": <int>,
             "budget_comment": "<short sentence>"
           },
           {
             "segment_type": "city_transfer",
             "from_city": "<string>",
             "to_city": "<string>",
             "detailed_mode": "<string>",
             "estimated_cost_jpy": <int>,
             "budget_comment": "<short sentence>",
             "gmaps_route": {
               "origin_city": "<string>",
               "dest_city": "<string>",
               "mode": "<string>",
               "duration_minutes": <int or null>,
               "distance_km": <float or null>,
               "summary": "<string or null>",
               "gmaps_directions_link": "<string>"
             }
           }
         ]
       },
       ...
     ]
   }

- Use only fields described above.
- Do NOT include explanations or commentary outside the JSON.
- The JSON must be syntactically valid.
""".strip()


def call_llm_make_itinerary(
    cfg: Config,
    qid: str,
    query_obj: Dict[str, Any],
    candidate_pois: List[Dict[str, Any]],
    city_scores: Dict[str, float],
    city_routes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Call the LLM "agent" with query, POIs, city_scores, city_routes, and
    intra-city distance hints, expecting a JSON itinerary.
    """

    # tighten candidate_pois to avoid giant prompts
    pois_for_prompt = candidate_pois[: cfg.max_pois_for_prompt]

    # build intra-city distance hints for these POIs
    intra_city_distance_hints = build_intra_city_distance_hints(pois_for_prompt)

    # Build a compact payload
    payload = {
        "query": {
            "qid": qid,
            "generated_query": query_obj.get("generated_query"),
            "total_days": query_obj.get("total_days"),
            "time_of_day_pref": query_obj.get("time_of_day_pref"),
            "duration_pref": query_obj.get("duration_pref"),
            "seasonality_pref": query_obj.get("seasonality_pref"),
            "atmosphere_pref": query_obj.get("atmosphere_pref"),
            "activity_pref": query_obj.get("activity_pref"),
            "budget_style": query_obj.get("budget_style"),
            "travel_companion_category": query_obj.get("travel_companion_category"),
            "round_trip": query_obj.get("round_trip"),
        },
        "candidate_pois": [
            {
                "poi_name": p.get("poi_name"),
                "city": p.get("city"),
                "score": p.get("score"),
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "types": p.get("types"),
                "landscape": p.get("landscape"),
                "activities": p.get("activities"),
                "atmosphere": p.get("atmosphere"),
                "season_primary": p.get("season_primary"),
                "season_secondary": p.get("season_secondary"),
                "time_of_day_counts": p.get("time_of_day_counts"),
                "duration_counts": p.get("duration_counts"),
            }
            for p in pois_for_prompt
        ],
        "cities": sorted(city_scores.keys()),
        "city_scores": city_scores,
        "city_routes": city_routes,
        "intra_city_distance_hints": intra_city_distance_hints,
    }

    payload_str = json.dumps(payload, ensure_ascii=False)

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Here is the full planning context as JSON.\n"
                "Use it to produce the final itinerary JSON with the exact required shape.\n\n"
                f"{payload_str}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=cfg.llm_model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content
    try:
        itinerary = json.loads(content)
    except json.JSONDecodeError as e:
        print("[ERROR] Failed to parse LLM JSON; raw content:")
        print(content)
        raise e

    return itinerary


# ========================
# Main orchestration
# ========================

def run_agent_planning(cfg: Config, qid: str) -> Dict[str, Any]:
    # Load data
    print(f"[INFO] Planning itinerary for QID={qid}")
    queries = load_queries_jsonl(cfg.user_queries_path)
    if qid not in queries:
        raise SystemExit(f"[ERROR] QID {qid!r} not found in {cfg.user_queries_path}")

    query_obj = queries[qid]

    poi_cards = load_poi_cards(cfg.poi_cards_path)
    topk_raw = load_topk_results(cfg.topk_dir, qid)

    # Build POI index and enrich top-k with lat/lon, etc.
    poi_index = build_poi_index(poi_cards)
    topk_enriched = attach_card_data_to_topk(topk_raw, poi_index)

    # Group by city and compute city scores
    city_to_pois = group_pois_by_city(topk_enriched)
    if not city_to_pois:
        raise SystemExit("[ERROR] No cities found in top-k; cannot plan.")

    city_scores = compute_city_scores(city_to_pois)
    cities = sorted(city_scores.keys())
    print(f"[INFO] Unique cities in top-k: {cities}")

    # Compute city centroids from the full POI DB (not used in GM yet, but available)
    centroids = compute_city_centroids_from_pois(poi_cards)

    # Build Google Maps city-to-city routes for the agent
    if len(cities) > 1:
        print("[INFO] Querying Google Maps for city-to-city routes...")
        city_routes = build_city_routes_for_agent(
            cfg,
            cities,
            centroids,
            query_obj.get("budget_style", "midrange"),
        )
    else:
        print("[INFO] Single-city itinerary; no inter-city transfers needed.")
        city_routes = []

    # Flatten top-k POIs into a single list; sort by score globally
    candidate_pois = sorted(topk_enriched, key=lambda x: x.get("score", 0.0), reverse=True)

    # Let the LLM agent design the final itinerary
    itinerary = call_llm_make_itinerary(
        cfg=cfg,
        qid=qid,
        query_obj=query_obj,
        candidate_pois=candidate_pois,
        city_scores=city_scores,
        city_routes=city_routes,
    )

    return itinerary


def main():
    parser = argparse.ArgumentParser(description="Agent-based multi-day planner from top-k POIs.")
    parser.add_argument("--qid", type=str, required=True, help="Query ID from user_queries_japan.jsonl")
    parser.add_argument("--topk-dir", type=str, default="./topk_pick", help="Directory containing topk_{qid}.json")
    parser.add_argument("--output-dir", type=str, default="./itinerary", help="Directory to write itinerary JSON")
    parser.add_argument("--max-pois-for-prompt", type=int, default=40, help="Max candidate POIs to show the agent")
    parser.add_argument("--model", type=str, default="gpt-5")

    args = parser.parse_args()

    cfg = Config(
        topk_dir=Path(args.topk_dir),
        output_dir=Path(args.output_dir),
        gmaps_api_key=os.getenv("GOOGLE_MAPS_API_KEY"),
        llm_model=args.model
    )
    cfg.max_pois_for_prompt = args.max_pois_for_prompt

    itinerary = run_agent_planning(cfg, args.qid)

    # Save itinerary
    out_path = cfg.output_dir / f"itinerary_agent_{args.qid}.json"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(itinerary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved itinerary → {out_path}")


if __name__ == "__main__":
    main()
