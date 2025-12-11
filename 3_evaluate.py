#!/usr/bin/env python3
"""
Itinerary evaluation script (single-file itinerary).

Usage example:

  # If your itinerary file is named like:
  #   itinerary/itinerary_agent_q00002.json
  # then run:
  #
  #   python 3_evaluate.py \
  #     --qid q00002 \
  #     --user-queries-path data/user_queries_japan.jsonl
  #
  # The script will automatically look for:
  #   itinerary/itinerary_agent_q00002.json
  # in the current working directory.
  #
  # The evaluation result will be written to:
  #   evaluate_result/evaluate_q00002.json

Notes:
- Hit Rate (HR) is judged by an LLM at the attraction level.
- Route / POI / semantic / overall feasibility are judged by LLMs.
- Distance information (when present) is only used internally (via edge
  distance lists) to help the LLM reason about routing and pacing; no numeric
  distance metrics are exposed in the final JSON.

Scales (all 1–10):
- route_feasibility_score:      1–10
- poi_feasibility_score:        1–10
- semantic_feasibility_score:   1–10
- overall_feasibility_score:    1–10
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


# ========================
# Config
# ========================

@dataclass
class Config:
    # poi_db_path is kept for CLI compatibility but not used
    poi_db_path: Path
    user_queries_path: Path
    itinerary_path: Path

    llm_model_relevance: str = "gpt-5-mini"
    llm_model_feasibility: str = "gpt-5-mini"

    relevance_batch_size: int = 12  # POIs per LLM batch


# ========================
# Simple distance helpers (internal only)
# ========================

def extract_city_transfer_distances(itinerary: Dict[str, Any]) -> List[float]:
    """
    Extract global distances (in km) from itinerary segments that contain
    a gmaps_route.distance_km field.

    We do NOT distinguish by city here; this is the whole itinerary path.
    This is used only as internal signal for LLMs (not exposed as metrics).
    """
    dists: List[float] = []
    for day in itinerary.get("days", []):
        for seg in day.get("segments", []):
            gr = seg.get("gmaps_route")
            if not isinstance(gr, dict):
                continue
            dk = gr.get("distance_km")
            if dk is None:
                continue
            try:
                dists.append(float(dk))
            except (TypeError, ValueError):
                continue
    return dists


def extract_city_transfer_distances_by_city(
    itinerary: Dict[str, Any]
) -> Dict[str, List[float]]:
    """
    Extract distances grouped by city.

    Heuristic:
    - For each segment with gmaps_route.distance_km, we try to assign it to
      a city bucket in this order:
        seg["city"] -> seg["from_city"] -> day["base_city"] -> "UNKNOWN"

    Used only for LLM reasoning, not exposed as metrics.
    """
    edges_by_city: Dict[str, List[float]] = {}

    for day in itinerary.get("days", []):
        base_city = (day.get("base_city") or "").strip() or None
        for seg in day.get("segments", []):
            gr = seg.get("gmaps_route")
            if not isinstance(gr, dict):
                continue
            dk = gr.get("distance_km")
            if dk is None:
                continue
            try:
                dkf = float(dk)
            except (TypeError, ValueError):
                continue

            seg_city = (
                (seg.get("city") or "")
                or (seg.get("from_city") or "")
                or (base_city or "")
            ).strip()
            if not seg_city:
                seg_city = "UNKNOWN"

            edges_by_city.setdefault(seg_city, []).append(dkf)

    return edges_by_city


# ========================
# Data loading
# ========================

def load_user_queries(cfg: Config) -> Dict[str, Dict[str, Any]]:
    """
    user_queries_japan.jsonl: one JSON object per line with key "qid".
    """
    out: Dict[str, Dict[str, Any]] = {}
    with cfg.user_queries_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[obj["qid"]] = obj
    return out


def load_itinerary(cfg: Config) -> Dict[str, Any]:
    """Load a single itinerary JSON file."""
    with cfg.itinerary_path.open() as f:
        return json.load(f)


# ========================
# Itinerary ⇒ attractions (for LLM HR)
# ========================

def extract_attractions_for_relevance(
    itinerary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Logical attraction list (for HR / LLM), no coord requirements.

    Each element:
      {
        "poi_name": ...,
        "city": ...,
        "landscape": [...],
        "activities": [...],
        "atmosphere": [...]
      }
    """
    atts: List[Dict[str, Any]] = []
    for day in itinerary.get("days", []):
        for seg in day.get("segments", []):
            if seg.get("segment_type") != "visit":
                continue
            city = (seg.get("city") or "").strip()
            poi_name = (seg.get("poi_name") or "").strip()
            if not city or not poi_name:
                continue
            atts.append(
                {
                    "poi_name": poi_name,
                    "city": city,
                    "landscape": seg.get("landscape", []),
                    "activities": seg.get("activities", []),
                    "atmosphere": seg.get("atmosphere", []),
                }
            )
    return atts


# ========================
# LLM-based metrics (HR + feasibility dimensions)
# ========================

RELEVANCE_SYSTEM_PROMPT = """
You are a strict travel relevance judge.

Given:
1) A user query describing a desired trip.
2) A list of attractions, each with city, name, landscape, activities, and atmosphere.

For EACH attraction, decide if it is SEMANTICALLY RELEVANT to the user's stated
preferences (destinations, budget style, seasonality, companion type, atmosphere,
and activities), regardless of exact route feasibility.

Respond ONLY as a JSON list of strings, one per attraction, each element
being exactly "RELEVANT" or "NOT_RELEVANT".
"""

ROUTE_FEASIBILITY_SYSTEM_PROMPT = """
You are an expert travel planner and logistics critic.

Your job is to evaluate the ROUTE FEASIBILITY of a multi-day itinerary for a given user query.

Focus on:
- Geographic sense of the route (order of cities, backtracking, weird zigzags).
- The pattern and magnitude of the distance edges in kilometers (when present).
- Whether the city-level distance distribution looks reasonable (per-city edge lists).
- Day-by-day pacing in terms of transfers vs. nights available.

You will receive JSON like:

{
  "user_query": {...},
  "itinerary": {...},
  "distance_stats_global": {
    "edges_km": [...]
  },
  "distance_stats_per_city": {
    "<city>": {
      "edges_km": [...]
    },
    ...
  }
}

Return ONLY a compact JSON object:
{
  "score": <integer from 1 to 10>,
  "explanation": "<short explanation>"
}

Interpretation (guideline):
  1–3  = clearly unrealistic / very inefficient,
  4–6  = somewhat feasible but with notable routing or pacing issues,
  7–10 = geographically sensible and realistic in pace and distances.

Keep the explanation to 1–3 sentences.
"""

POI_FEASIBILITY_SYSTEM_PROMPT = """
You are an expert on on-the-ground travel logistics and attraction planning.

Evaluate the POI FEASIBILITY of an itinerary for a given user query.

Focus on:
- Whether the chosen POIs per day could realistically be visited given travel
  distances in the itinerary (you may see edge distances where available).
- Obvious time conflicts: too many far-apart spots in one day, unrealistic chaining.
- Reasonable sequencing by time of day (e.g., not putting late-night bars at 9am).
- Plausibility of visit density for typical travelers with the given companion type.

You will receive JSON like:

{
  "user_query": {...},
  "itinerary": {...}
}

Return ONLY a compact JSON object:
{
  "score": <integer from 1 to 10>,
  "explanation": "<short explanation>"
}

Interpretation (guideline):
  1–3  = POI packing clearly impossible / absurd,
  4–6  = mixed, some days overstuffed or awkward,
  7–10 = day-by-day POI choices look quite realistic and visitable.

Explain in 1–3 sentences.
"""

SEMANTIC_FEASIBILITY_SYSTEM_PROMPT = """
You are a semantic alignment judge for travel itineraries.

Evaluate how well the itinerary matches the user's stated preferences
(directions, cities/regions, seasonality, budget style, travel companion,
atmosphere, activities, time-of-day/duration preferences).

You will receive JSON like:

{
  "user_query": {...},
  "attractions": [
    {
      "city": ...,
      "poi_name": ...,
      "landscape": [...],
      "activities": [...],
      "atmosphere": [...]
    },
    ...
  ],
  "hit_rate": <float between 0 and 1>
}

The hit_rate is the fraction of attractions previously classified as relevant.

Return ONLY a compact JSON object:
{
  "score": <integer from 1 to 10>,
  "explanation": "<short explanation>"
}

Interpretation (guideline):
  1–3  = mostly off from the user's described trip,
  4–6  = partially aligned but missing or misinterpreting key aspects,
  7–10 = strongly matches the user's desired style, places, and constraints.

Mention both the qualitative match and how the hit_rate feels.
"""

OVERALL_FEASIBILITY_SYSTEM_PROMPT = """
You are an expert travel planner giving a SINGLE OVERALL FEASIBILITY judgment.

Combine:
- Semantic match with the user query (including hit_rate and semantic_feasibility_score).
- Route realism (route_feasibility_score and the overall pattern of distances).
- POI-level realism (poi_feasibility_score).
- Any obvious inconsistencies in days, transfers, or pace.

You will receive JSON like:

{
  "user_query": {...},
  "itinerary": {...},
  "distance_stats_global": {...},
  "distance_stats_per_city": {...},
  "hit_rate": <float>,
  "sub_scores": {
    "route_feasibility_score": <int>,
    "poi_feasibility_score": <int>,
    "semantic_feasibility_score": <int>
  }
}

Return ONLY a compact JSON object:
{
  "score": <integer from 1 to 10>,
  "explanation": "<short explanation>"
}

Interpretation (guideline):
  1–3  = very poor overall match / realism,
  4–6  = mixed, with significant issues,
  7–8  = good overall alignment and logistics with minor issues,
  9–10 = excellent overall alignment AND logistical soundness.

In your explanation (1–3 sentences), briefly mention:
- semantic match,
- routing/logistics,
- POI/day realism.
"""


def batch_llm_relevance(
    client: OpenAI,
    cfg: Config,
    query_obj: Dict[str, Any],
    attractions: List[Dict[str, Any]],
) -> List[bool]:
    """Return relevance flags (True/False) for each attraction (LLM-judged)."""
    payloads = []
    for a in attractions:
        payloads.append(
            {
                "city": a["city"],
                "poi_name": a["poi_name"],
                "landscape": a.get("landscape", []),
                "activities": a.get("activities", []),
                "atmosphere": a.get("atmosphere", []),
            }
        )

    labels_all: List[str] = []
    for i in range(0, len(payloads), cfg.relevance_batch_size):
        batch = payloads[i : i + cfg.relevance_batch_size]
        user_content = json.dumps(
            {"user_query": query_obj, "attractions": batch},
            ensure_ascii=False,
        )

        resp = client.chat.completions.create(
            model=cfg.llm_model_relevance,
            messages=[
                {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
        content = resp.choices[0].message.content
        try:
            labels = json.loads(content)
        except Exception:
            labels = ["NOT_RELEVANT"] * len(batch)

        if len(labels) != len(batch):
            labels = ["NOT_RELEVANT"] * len(batch)

        labels_all.extend(labels)

    return [lab.strip().upper() == "RELEVANT" for lab in labels_all]


def compute_hit_rate(
    client: OpenAI,
    cfg: Config,
    query_obj: Dict[str, Any],
    attractions: List[Dict[str, Any]],
) -> float:
    """
    HR: fraction of attractions labeled RELEVANT by the LLM.
    """
    if not attractions:
        return 0.0
    flags = batch_llm_relevance(client, cfg, query_obj, attractions)
    hits = sum(1 for f in flags if f)
    return hits / len(attractions)


def _generic_feasibility_call(
    client: OpenAI,
    model: str,
    system_prompt: str,
    payload: Dict[str, Any],
) -> Tuple[int, str]:
    """Helper to call an LLM that returns {score, explanation} JSON."""
    user_payload = json.dumps(payload, ensure_ascii=False)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
    )
    content = resp.choices[0].message.content
    try:
        obj = json.loads(content)
        score = int(obj.get("score", 0))
        explanation = str(obj.get("explanation", "")).strip()
    except Exception:
        score = 0
        explanation = "Failed to parse score."
    return score, explanation


def compute_route_feasibility_score(
    client: OpenAI,
    cfg: Config,
    query_obj: Dict[str, Any],
    itinerary: Dict[str, Any],
    distance_stats_global: Dict[str, Any],
    distance_stats_per_city: Dict[str, Any],
) -> Tuple[int, str]:
    payload = {
        "user_query": query_obj,
        "itinerary": itinerary,
        "distance_stats_global": distance_stats_global,
        "distance_stats_per_city": distance_stats_per_city,
    }
    return _generic_feasibility_call(
        client,
        cfg.llm_model_feasibility,
        ROUTE_FEASIBILITY_SYSTEM_PROMPT,
        payload,
    )


def compute_poi_feasibility_score(
    client: OpenAI,
    cfg: Config,
    query_obj: Dict[str, Any],
    itinerary: Dict[str, Any],
) -> Tuple[int, str]:
    payload = {
        "user_query": query_obj,
        "itinerary": itinerary,
    }
    return _generic_feasibility_call(
        client,
        cfg.llm_model_feasibility,
        POI_FEASIBILITY_SYSTEM_PROMPT,
        payload,
    )


def compute_semantic_feasibility_score(
    client: OpenAI,
    cfg: Config,
    query_obj: Dict[str, Any],
    attractions: List[Dict[str, Any]],
    hr: float,
) -> Tuple[int, str]:
    payload = {
        "user_query": query_obj,
        "attractions": attractions,
        "hit_rate": hr,
    }
    return _generic_feasibility_call(
        client,
        cfg.llm_model_feasibility,
        SEMANTIC_FEASIBILITY_SYSTEM_PROMPT,
        payload,
    )


def compute_overall_feasibility_score(
    client: OpenAI,
    cfg: Config,
    query_obj: Dict[str, Any],
    itinerary: Dict[str, Any],
    distance_stats_global: Dict[str, Any],
    distance_stats_per_city: Dict[str, Any],
    hr: float,
    route_score: int,
    poi_score: int,
    semantic_score: int,
) -> Tuple[int, str]:
    payload = {
        "user_query": query_obj,
        "itinerary": itinerary,
        "distance_stats_global": distance_stats_global,
        "distance_stats_per_city": distance_stats_per_city,
        "hit_rate": hr,
        "sub_scores": {
            "route_feasibility_score": route_score,
            "poi_feasibility_score": poi_score,
            "semantic_feasibility_score": semantic_score,
        },
    }
    return _generic_feasibility_call(
        client,
        cfg.llm_model_feasibility,
        OVERALL_FEASIBILITY_SYSTEM_PROMPT,
        payload,
    )


# ========================
# Driver
# ========================

def evaluate(cfg: Config, qid: str) -> Dict[str, Any]:
    client = OpenAI()

    queries = load_user_queries(cfg)
    query_obj = queries[qid]
    itinerary = load_itinerary(cfg)

    # Attractions for HR / LLM
    attractions = extract_attractions_for_relevance(itinerary)

    # Distance metrics: global + per-city (internal only)
    edges_km = extract_city_transfer_distances(itinerary)
    city_edges = extract_city_transfer_distances_by_city(itinerary)

    distance_stats_global = {
        "edges_km": edges_km,
    }

    distance_stats_per_city: Dict[str, Dict[str, Any]] = {}
    for city, edges in city_edges.items():
        distance_stats_per_city[city] = {
            "edges_km": edges,
        }

    # LLM-based metrics
    hr = compute_hit_rate(client, cfg, query_obj, attractions)

    route_score, route_expl = compute_route_feasibility_score(
        client,
        cfg,
        query_obj,
        itinerary,
        distance_stats_global,
        distance_stats_per_city,
    )

    poi_score, poi_expl = compute_poi_feasibility_score(
        client,
        cfg,
        query_obj,
        itinerary,
    )

    semantic_score, semantic_expl = compute_semantic_feasibility_score(
        client,
        cfg,
        query_obj,
        attractions,
        hr,
    )

    overall_score, overall_expl = compute_overall_feasibility_score(
        client,
        cfg,
        query_obj,
        itinerary,
        distance_stats_global,
        distance_stats_per_city,
        hr,
        route_score,
        poi_score,
        semantic_score,
    )

    # Main result object
    return {
        "qid": qid,
        "num_attractions": len(attractions),
        "num_travel_edges": len(edges_km),

        # Semantic relevance
        "HR": hr,

        # LLM scores (all 1–10)
        "route_feasibility_score": route_score,
        "route_feasibility_explanation": route_expl,
        "poi_feasibility_score": poi_score,
        "poi_feasibility_explanation": poi_expl,
        "semantic_feasibility_score": semantic_score,
        "semantic_feasibility_explanation": semantic_expl,
        "overall_feasibility_score": overall_score,
        "overall_feasibility_explanation": overall_expl,
    }


# ========================
# Auto itinerary path helper
# ========================

def auto_itinerary_path(qid: str) -> Path:
    """
    Automatically construct itinerary filename from QID.

    Examples:
      qid = "q00002" -> "itinerary/itinerary_agent_q00002.json"
      qid = "Q8"     -> "itinerary/itinerary_agent_q8.json"

    We normalize to lowercase for the filename, but keep the original
    qid string for looking up in user_queries.
    """
    qid_norm = qid.strip().lower()
    return Path(f"itinerary/itinerary_agent_{qid_norm}.json")


# ========================
# Main
# ========================

def main():
    parser = argparse.ArgumentParser(description="Evaluate a single itinerary JSON file.")
    parser.add_argument("--qid", required=True, help="Query ID, e.g. q00002")

    parser.add_argument(
        "--poi-db-path",
        type=Path,
        default=Path("data/poi_cards_clean_structured.json"),
        help="(Unused) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--user-queries-path",
        type=Path,
        default=Path("data/user_queries_japan.jsonl"),
        help="Path to user queries JSONL.",
    )

    args = parser.parse_args()

    # Auto-generate itinerary path from qid
    itinerary_path = auto_itinerary_path(args.qid)

    if not itinerary_path.exists():
        print(f"[WARN] Auto-generated itinerary file does not exist: {itinerary_path}")

    cfg = Config(
        poi_db_path=args.poi_db_path,      # not used
        user_queries_path=args.user_queries_path,
        itinerary_path=itinerary_path,
    )

    result = evaluate(cfg, args.qid)

    # ===== Write evaluation result to evaluate_result/ =====
    out_dir = Path("evaluate_result")
    out_dir.mkdir(parents=True, exist_ok=True)
    qid_norm = args.qid.strip().lower()
    out_path = out_dir / f"evaluate_{qid_norm}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Also print to stdout for quick inspection
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[INFO] Saved evaluation result to: {out_path}")


if __name__ == "__main__":
    main()
