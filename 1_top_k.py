#!/usr/bin/env python3
"""
Model-based POI matcher using precomputed POI embeddings.

- Reads:
    poi_cards_clean_structured.json        (list of POI cards)
    poi_clean_emb_cache.json               (precomputed POI embeddings)
    user_queries_japan.jsonl         (one query per line, with "qid")
- Uses:
    OpenAI text-embedding-3-large to embed QUERIES only
- Returns:
    Top-k POIs ranked by cosine similarity.

Supports:
    - Single query: --qid Q5
    - Batch queries Q1–Q20: --batch

Filtering behavior:
    1) We filter out POIs whose names are essentially just city-level labels,
       e.g. "Osaka", "Osaka (general)", "Nagano (day trip)".
    2) We then use an LLM to drop items that are primarily:
       - transportation segments (shinkansen ride, boat ride, flight, airport, etc.)
       - pure accommodation (hotel / ryokan / hostel / guesthouse etc.)
    Only *real visitable POIs* (temples, shrines, parks, museums, neighborhoods,
    viewpoints, festivals, etc.) are returned in the top-k results.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm


# ========================
# Config
# ========================

@dataclass
class Config:
    poi_cards_path: Path = Path("data/poi_cards_clean_structured.json")
    user_queries_path: Path = Path("data/user_queries_japan.jsonl")
    poi_emb_cache_path: Path = Path("data/poi_clean_emb_cache.json")

    embed_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-5-mini"   # for POI vs transport/hotel classification
    top_k: int = 20                 # default top-k


client = OpenAI()


# ========================
# Helpers: text builders
# ========================

def build_query_text(query: Dict[str, Any]) -> str:
    """
    Construct a textual representation of the user query that aligns
    with the POI fields we care about.
    """
    gen_q = query.get("generated_query", "")

    activities = query.get("activity_pref", [])
    if isinstance(activities, str):
        activities_str = activities
    else:
        activities_str = "; ".join(activities)

    atmosphere = query.get("atmosphere_pref", "")
    season = query.get("seasonality_pref", "")
    time_of_day = query.get("time_of_day_pref", "")
    duration = query.get("duration_pref", "")

    text = (
        f"User wants an itinerary in Japan. "
        f"Original query: {gen_q} "
        f"Preferred activities: {activities_str}. "
        f"Preferred atmosphere: {atmosphere}. "
        f"Preferred season: {season}. "
        f"Preferred time of day: {time_of_day}. "
        f"Preferred visit duration: {duration}."
    )

    return text


# ========================
# Embedding + similarity
# ========================

def embed_query(text: str, model: str) -> List[float]:
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ========================
# Data loading
# ========================

def load_poi_cards(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text())


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


def load_poi_embeddings(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"POI embedding cache {path} not found. "
            f"Run build_poi_embeddings.py first."
        )
    return json.loads(path.read_text())


# ========================
# City-ish POI detector (deterministic)
# ========================

GENERIC_CITY_SUBSTRINGS = [
    "(general",
    "(day trip",
    "(day/night trip",
    "day trip)",
    "day/night trip)",
    "city /",
    "city)",
    "city -",
    "city base",
    "city stay",
    "city visit",
    "city overall",
    "onsen town",
    "overnight stay",
    "brief stay",
    "base for",
    "base town",
]


def is_cityish_poi(name: str, city: str) -> bool:
    """
    Return True if this POI name is basically a city-level label, not a specific POI.

    Rules:
      - exact match with city name => True
      - starts with city name and then adds only generic qualifiers like:
           "(general)", "(day trip)", "(day/night trip)", "onsen town / ryokan stay", etc.
    """
    if not name or not city:
        return False

    n = name.strip().lower()
    c = city.strip().lower()

    # Exact match, e.g. "Osaka"
    if n == c:
        return True

    # Names that start with the city and then only add generic "fluff"
    if n.startswith(c):
        for bad in GENERIC_CITY_SUBSTRINGS:
            if bad in n:
                return True

    # Special generic multi-city labels
    if "tokyo-kyoto-osaka" in n or "tokyo kyoto osaka" in n:
        return True

    return False


# ========================
# LLM-based POI vs transport/hotel classifier
# ========================

POI_CLASSIFIER_SYSTEM = """
You are helping clean a travel POI database.

For each candidate, decide if it is a GOOD "visit POI" for an itinerary,
or if it is primarily LOGISTICS (transport / hotel / generic base) that
should be excluded from the top-k POI ranking.

GOOD "visit POIs" (KEEP):
- specific temples, shrines, gardens, parks, museums, towers, viewpoints
- specific neighborhoods/districts/streets used as sightseeing (e.g., Asakusa, Shinjuku, Dotonbori)
- specific attractions in theme parks (e.g., "Tower of Terror (DisneySea)")
- specific festivals or seasonal events
- specific scenic lakes, bridges, islands, villages, tea fields, etc.

BAD logistics (DROP):
- train/shinkansen rides, flights, bus rides, boat rides
- airports, stations, ferry terminals
- hotels, ryokan, hostels, guesthouses or "overnight stays" that are mainly accommodation
- generic "base" towns used as just where to sleep
- generic city labels already filtered elsewhere (but if you see them, also DROP)

You will be given:
- poi_name
- city
- types (a rough category list)
- landscape (description)
- activities (what the traveler does there)

TASK:
- Answer KEEP if this is a real visit POI / attraction a traveler might plan to *experience*.
- Answer DROP if this is mainly transport, hotel/accommodation, airport, or generic city/base.

RESPOND WITH EXACTLY ONE TOKEN:
KEEP
or
DROP

No explanations.
""".strip()


def llm_should_keep_poi(cfg: Config, poi: Dict[str, Any], cache: Dict[Tuple[str, str, str], bool]) -> bool:
    """
    Use LLM to decide if this POI is a real visitable attraction (KEEP)
    or primarily transport/hotel/logistics (DROP).

    cache key: (poi_name.lower(), city.lower(), types_string.lower())
    """
    poi_name = (poi.get("poi_name") or "").strip()
    city = (poi.get("city") or "").strip()

    types = poi.get("types") or []
    if isinstance(types, list):
        types_str = ", ".join(t for t in types if t)
    else:
        types_str = str(types)

    landscape = poi.get("landscape") or []
    if isinstance(landscape, list):
        landscape_str = "; ".join(str(x) for x in landscape if x)
    else:
        landscape_str = str(landscape)

    activities = poi.get("activities") or []
    if isinstance(activities, list):
        activities_str = "; ".join(str(x) for x in activities if x)
    else:
        activities_str = str(activities)

    key: Tuple[str, str, str] = (
        poi_name.lower(),
        city.lower(),
        types_str.lower(),
    )

    if key in cache:
        return cache[key]

    user_content = (
        f'poi_name: "{poi_name}"\n'
        f'city: "{city}"\n'
        f'types: "{types_str}"\n'
        f'landscape: "{landscape_str}"\n'
        f'activities: "{activities_str}"\n\n'
        "Should this be kept as a VISIT POI (KEEP) or dropped as logistics (DROP)?"
    )

    try:
        resp = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[
                {"role": "system", "content": POI_CLASSIFIER_SYSTEM},
                {"role": "user", "content": user_content},
            ]
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
    except Exception as e:
        # If classification fails, be conservative and KEEP (so we don't over-drop)
        print(f"[WARN] LLM POI classifier failed for {poi_name!r} in {city!r}: {e}")
        cache[key] = True
        return True

    keep = (answer == "KEEP")
    cache[key] = keep
    return keep


# ========================
# Matching (with filters)
# ========================

def rank_pois_for_query(
    cfg: Config,
    query: Dict[str, Any],
    pois: List[Dict[str, Any]],
    poi_embs: List[Dict[str, Any]],
    city_cache: Dict[int, bool],  # kept for compatibility (not used)
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Given a query + POIs + POI embeddings, return up to top_k ranked results.

    Filters:
      1) Skip city-ish POIs (is_cityish_poi).
      2) Use LLM to DROP transport/hotel/logistics items.
    """
    if top_k is None:
        top_k = cfg.top_k

    query_text = build_query_text(query)
    q_emb = embed_query(query_text, cfg.embed_model)

    n_pois = len(pois)

    # 1) score all valid POIs
    scored: List[Dict[str, Any]] = []
    for item in poi_embs:
        idx = item.get("poi_index")

        # sanity check index
        if not isinstance(idx, int) or idx < 0 or idx >= n_pois:
            # stale embedding entry
            continue

        poi = pois[idx]
        score = cosine(q_emb, item["embedding"])

        scored.append(
            {
                "poi_index": idx,
                "poi_item": item,
                "score": score,
            }
        )

    # 2) sort by similarity
    scored.sort(key=lambda x: x["score"], reverse=True)

    # LLM per-POI cache
    poi_keep_cache: Dict[Tuple[str, str, str], bool] = {}

    # 3) walk the sorted list, applying filters
    results: List[Dict[str, Any]] = []

    for entry in scored:
        if len(results) >= top_k:
            break

        idx = entry["poi_index"]
        poi = pois[idx]

        poi_name = (poi.get("poi_name") or "").strip()
        city = (poi.get("city") or "").strip()

        # 3a) Hard city-ish filter
        if is_cityish_poi(poi_name, city):
            # Uncomment for debugging:
            # print(f"[DEBUG] Skipping city-ish POI idx={idx}: {poi_name!r} in {city!r}")
            continue

        # 3b) LLM: keep only real visit POIs (not transport/hotel)
        if not llm_should_keep_poi(cfg, poi, poi_keep_cache):
            # Uncomment for debugging:
            # print(f"[DEBUG] Dropping logistics POI idx={idx}: {poi_name!r} in {city!r}")
            continue

        item = entry["poi_item"]

        results.append(
            {
                "poi": poi,
                "poi_name": poi_name,
                "city": city,
                "score": entry["score"],
            }
        )

    return results


# ========================
# Batch helper
# ========================

def run_batch_queries(
    cfg: Config,
    pois: List[Dict[str, Any]],
    poi_embs: List[Dict[str, Any]],
    queries: Dict[str, Dict[str, Any]],
    city_cache: Dict[int, bool],
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    """
    Generate model-matched POIs for qids in [start, end] and return as JSON list.
    Assumes QIDs are formatted like 'Q1', 'Q2', ... 'Q20'.
    """
    results: List[Dict[str, Any]] = []

    for i in tqdm(range(start, end + 1), desc="Batch queries"):
        qid = f"Q{i}"

        if qid not in queries:
            print(f"[WARN] QID {qid} not found, skipping...")
            continue

        query = queries[qid]
        ranked = rank_pois_for_query(cfg, query, pois, poi_embs, city_cache, top_k=cfg.top_k)

        results.append(
            {
                "qid": qid,
                "generated_query": query.get("generated_query"),
                "top_pois": [
                    {
                        "poi_name": r["poi_name"],
                        "city": r["city"],
                        "score": r["score"],
                        "types": r["poi"].get("types"),
                        "landscape": r["poi"].get("landscape"),
                        "activities": r["poi"].get("activities"),
                        "atmosphere": r["poi"].get("atmosphere"),
                        "season_primary": r["poi"].get("season_primary"),
                        "season_secondary": r["poi"].get("season_secondary"),
                    }
                    for r in ranked
                ],
            }
        )

    return results


# ========================
# CLI / main
# ========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model-based POI matcher using precomputed POI embeddings.")
    parser.add_argument("--qid", type=str, help="Single query ID from user_queries_japan.jsonl (e.g. Q5)")
    parser.add_argument("--batch", action="store_true", help="Run batch mode for QIDs Q<start>–Q<end> (ignore --qid)")
    parser.add_argument("--start", type=int, default=1, help="Batch start (default Q1)")
    parser.add_argument("--end", type=int, default=20, help="Batch end (default Q20)")
    parser.add_argument("--topk", type=int, default=40, help="Top-k POIs to return")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--model", type=str, default="gpt-5")
    args = parser.parse_args()

    cfg = Config()
    cfg.top_k = args.topk
    cfg.llm_model = args.model

    pois = load_poi_cards(cfg.poi_cards_path)
    poi_embs = load_poi_embeddings(cfg.poi_emb_cache_path)
    queries = load_queries_jsonl(cfg.user_queries_path)

    # quick sanity check: stale embedding indices
    max_idx = max(e.get("poi_index", -1) for e in poi_embs)
    if max_idx >= len(pois):
        print(
            f"[WARN] poi_emb_cache contains indices up to {max_idx}, "
            f"but there are only {len(pois)} POIs. Some stale entries will be ignored."
        )

    # kept for compatibility, but not used directly in current filters
    city_cache: Dict[int, bool] = {}

    # -------- Batch mode --------
    if args.batch:
        batch_results = run_batch_queries(
            cfg,
            pois,
            poi_embs,
            queries,
            city_cache,
            start=args.start,
            end=args.end,
        )

        out_path = Path(f"topk_batch_Q{args.start}_Q{args.end}.json")
        out_path.write_text(
            json.dumps(batch_results, indent=2 if args.pretty else None, ensure_ascii=False)
        )
        print(f"[INFO] Saved batch results → {out_path}")

        if args.pretty:
            print(json.dumps(batch_results, indent=2, ensure_ascii=False))

    else:
        # -------- Single-query mode --------
        if not args.qid:
            raise SystemExit("Either specify --qid for single query or use --batch mode.")

        if args.qid not in queries:
            raise SystemExit(f"QID {args.qid!r} not found in {cfg.user_queries_path}.")

        query = queries[args.qid]
        ranked = rank_pois_for_query(cfg, query, pois, poi_embs, city_cache, top_k=cfg.top_k)

        output: List[Dict[str, Any]] = []
        for r in ranked:
            poi = r["poi"]
            output.append(
                {
                    "poi_name": r["poi_name"],
                    "city": r["city"],
                    "score": r["score"],
                    "types": poi.get("types"),
                    "landscape": poi.get("landscape"),
                    "activities": poi.get("activities"),
                    "atmosphere": poi.get("atmosphere"),
                    "season_primary": poi.get("season_primary"),
                    "season_secondary": poi.get("season_secondary"),
                }
            )

        out_path = Path(f"topk_pick/topk_{args.qid}.json")
        out_path.write_text(
            json.dumps(output, indent=2 if args.pretty else None, ensure_ascii=False)
        )
        print(f"[INFO] Saved single-query results → {out_path}")

        if args.pretty:
            print(json.dumps(output, indent=2, ensure_ascii=False))
