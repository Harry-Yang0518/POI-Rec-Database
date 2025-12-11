#!/usr/bin/env python3
"""
Run the full POI → itinerary → evaluation pipeline for a range of queries.

Pipeline per query:
  Stage 0: 0_gen_user_query.py  (once, to generate N queries)
  Stage 1: 1_top_k.py           (per qid)
  Stage 2: 2_plan_route.py      (per qid)
  Stage 3: 3_evaluate.py        (per qid)

You can now choose which stages to run:

Examples:

  # Full pipeline: generate 100 queries, then run 1→2→3 for all of them
  python run_full_pipeline.py --num-queries 100 --stages "0,1,2,3"

  # Only generate 100 queries (no downstream processing)
  python run_full_pipeline.py --num-queries 100 --stages "0"

  # Assume queries already exist; run only stages 1–3 for qids 11..20
  python run_full_pipeline.py --num-queries 100 --start-idx 11 --end-idx 20 --stages "1,2,3"

  # Only re-run evaluation (stage 3) for first 50 qids
  python run_full_pipeline.py --num-queries 50 --stages "3"

Assumptions:
- 0_gen_user_query.py writes all queries to data/user_queries_japan.jsonl
- Each line in that file is a JSON object with a "qid" field, e.g. "q00010"
- Scripts:
    0_gen_user_query.py
    1_top_k.py
    2_plan_route.py
    3_evaluate.py
  are in the same directory as this script (or on PATH).
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Set


def run_cmd(cmd: List[str]) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def generate_user_queries(num_queries: int, user_queries_path: Path) -> None:
    """
    Call 0_gen_user_query.py once to generate all queries (Stage 0).
    """
    print(f"[STEP] Generating {num_queries} user queries (Stage 0)...")
    run_cmd([
        "python",
        "0_gen_user_query.py",
        "--num-queries",
        str(num_queries),
    ])

    if not user_queries_path.exists():
        raise FileNotFoundError(
            f"Expected user queries at {user_queries_path}, "
            "but file does not exist after generation."
        )


def load_all_qids(
    user_queries_path: Path,
    limit: Optional[int] = None
) -> List[str]:
    """
    Read qids from user_queries_japan.jsonl, preserving order.
    """
    print(f"[STEP] Loading qids from {user_queries_path}...")
    if not user_queries_path.exists():
        raise FileNotFoundError(
            f"User queries file not found at {user_queries_path}. "
            "If you did not run Stage 0 in this invocation, make sure "
            "the file exists from a previous run."
        )

    qids: List[str] = []
    with user_queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qid")
            if not qid:
                continue
            qids.append(qid)
            if limit is not None and len(qids) >= limit:
                break

    print(f"[INFO] Loaded {len(qids)} qids.")
    return qids


def run_pipeline_for_qid(qid: str, stages: Set[int]) -> None:
    """
    Run any subset of (1_top_k, 2_plan_route, 3_evaluate) for a single qid.
    """
    print(f"\n[PIPELINE] Processing {qid} with stages {sorted(stages)} ...")

    # Stage 1: top-k POIs
    if 1 in stages:
        print(f"[STAGE 1] 1_top_k.py for {qid}")
        run_cmd([
            "python",
            "1_top_k.py",
            "--qid",
            qid,
        ])

    # Stage 2: plan multi-day route
    if 2 in stages:
        print(f"[STAGE 2] 2_plan_route.py for {qid}")
        run_cmd([
            "python",
            "2_plan_route.py",
            "--qid",
            qid,
        ])

    # Stage 3: evaluate itinerary
    if 3 in stages:
        print(f"[STAGE 3] 3_evaluate.py for {qid}")
        run_cmd([
            "python",
            "3_evaluate.py",
            "--qid",
            qid,
        ])

    print(f"[DONE] Finished stages {sorted(stages)} for {qid}")


def parse_stages(stages_str: str) -> Set[int]:
    """
    Parse a string like "0,1,2,3" or "1 3" into a set of integers {0,1,2,3}.
    """
    parts = stages_str.replace(",", " ").split()
    stages: Set[int] = set()
    for p in parts:
        if not p:
            continue
        try:
            val = int(p)
        except ValueError:
            raise ValueError(f"Invalid stage value '{p}'. Must be 0, 1, 2, or 3.")
        if val not in (0, 1, 2, 3):
            raise ValueError(f"Invalid stage value '{p}'. Must be 0, 1, 2, or 3.")
        stages.add(val)

    if not stages:
        raise ValueError("No stages specified. Must choose at least one of 0,1,2,3.")
    return stages


def main():
    parser = argparse.ArgumentParser(
        description="Run 0→1→2→3 itinerary pipeline for a range of queries."
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help=(
            "Number of user queries to generate (for Stage 0) and/or the "
            "maximum number of qids to load from the queries JSONL."
        ),
    )
    parser.add_argument(
        "--user-queries-path",
        type=Path,
        default=Path("data/user_queries_japan.jsonl"),
        help="Path to user queries JSONL file.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=1,
        help=(
            "1-based start index (in the JSONL order) of queries to run. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help=(
            "1-based end index (inclusive) of queries to run. "
            "Default: None (use up to num-queries or available qids)."
        ),
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="0,1,2,3",
        help=(
            "Which stages to run, as a comma/space-separated subset of 0,1,2,3.\n"
            "  0 = generate user queries (0_gen_user_query.py; once)\n"
            "  1 = top-k POIs per qid (1_top_k.py)\n"
            "  2 = route planning per qid (2_plan_route.py)\n"
            "  3 = evaluation per qid (3_evaluate.py)\n"
            "Example: --stages '0,1,2,3' (default full pipeline), "
            "--stages '1,2,3', --stages '3', etc."
        ),
    )
    args = parser.parse_args()

    try:
        stages = parse_stages(args.stages)
    except ValueError as e:
        parser.error(str(e))

    # Stage 0: generate queries if requested
    if 0 in stages:
        generate_user_queries(args.num_queries, args.user_queries_path)
    else:
        print("[INFO] Stage 0 not requested; will not generate new user queries.")

    # Determine if we need to run any per-qid stages
    per_qid_stages = {s for s in stages if s in (1, 2, 3)}
    if not per_qid_stages:
        print("[INFO] No per-qid stages (1/2/3) requested. Done.")
        return

    # Load qids from JSONL (use up to num-queries qids)
    qids = load_all_qids(args.user_queries_path, limit=args.num_queries)
    total_q = len(qids)
    if total_q == 0:
        print("[WARN] No qids loaded; nothing to do.")
        return

    # Clamp start/end indices
    start_idx = max(1, args.start_idx)
    end_idx = args.end_idx if args.end_idx is not None else total_q
    end_idx = min(end_idx, total_q)

    if start_idx > end_idx:
        print(
            f"[WARN] Invalid range: start-idx ({start_idx}) > end-idx ({end_idx}). "
            "Nothing will be run."
        )
        return

    # Convert to 0-based slice
    qids_range = qids[start_idx - 1:end_idx]
    print(
        f"[INFO] Running stages {sorted(per_qid_stages)} "
        f"for indices {start_idx}..{end_idx} "
        f"({len(qids_range)} queries)."
    )

    # Run selected stages per qid in that range
    for offset, qid in enumerate(qids_range, start=start_idx):
        print(f"\n========== [{offset}/{end_idx}] {qid} ==========")
        try:
            run_pipeline_for_qid(qid, per_qid_stages)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed for {qid}: {e}")
            # Continue to next qid
            continue


if __name__ == "__main__":
    main()
