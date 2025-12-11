#!/usr/bin/env python3
"""
Aggregate and analyze itinerary evaluation results, and save the summary to a file.

Usage:

  python analyze_evals.py \
      --results-dir evaluate_result \
      --output-file evaluation_summary.txt
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------
# Tee Printer (console + file)
# ---------------------------
class Tee:
    def __init__(self, filepath: Path):
        self.file = filepath.open("w", encoding="utf-8")

    def write(self, msg: str):
        print(msg, end="")       # console
        self.file.write(msg)     # file

    def close(self):
        self.file.close()


# ---------------------------
# Utility: safe numeric getter
# ---------------------------
def safe_get(d: Dict[str, Any], key: str, default=None):
    v = d.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


# ---------------------------
# Load all evaluation JSONs
# ---------------------------
def load_results(results_dir: Path):
    results = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            obj["_filename"] = path.name
            results.append(obj)
        except Exception as e:
            print(f"[WARN] Failed to parse {path}: {e}")
    return results


# ---------------------------
# Summary statistics
# ---------------------------
def stat_block(name: str, values: List[float], tee: Tee):
    if not values:
        tee.write(f"- {name}: no data\n")
        return
    mean = statistics.mean(values)
    median = statistics.median(values)
    minimum = min(values)
    maximum = max(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0

    tee.write(
        f"- {name}: "
        f"mean={mean:.3f}, "
        f"median={median:.3f}, "
        f"min={minimum:.3f}, "
        f"max={maximum:.3f}, "
        f"std={stdev:.3f}\n"
    )


def summarize(results: List[Dict[str, Any]], tee: Tee):
    n = len(results)
    tee.write(f"Loaded {n} evaluation results.\n\n")

    hrs, rfs, pfs, sfs, ofs = [], [], [], [], []

    for r in results:
        hr = safe_get(r, "HR")
        rf = safe_get(r, "route_feasibility_score")
        pf = safe_get(r, "poi_feasibility_score")
        sf = safe_get(r, "semantic_feasibility_score")
        of = safe_get(r, "overall_feasibility_score")

        if hr is not None: hrs.append(hr)
        if rf is not None: rfs.append(rf)
        if pf is not None: pfs.append(pf)
        if sf is not None: sfs.append(sf)
        if of is not None: ofs.append(of)

    tee.write("=== Global Summary ===\n")
    stat_block("Hit Rate (HR)", hrs, tee)
    stat_block("Route feasibility", rfs, tee)
    stat_block("POI feasibility", pfs, tee)
    stat_block("Semantic feasibility", sfs, tee)
    stat_block("Overall feasibility", ofs, tee)
    tee.write("\n")


# ---------------------------
# Top/bottom K
# ---------------------------
def print_top_bottom(results, k, tee: Tee):
    scored = []
    for r in results:
        of = safe_get(r, "overall_feasibility_score")
        if of is not None:
            scored.append((of, r))

    if not scored:
        tee.write("No scores found.\n")
        return

    scored.sort(key=lambda x: x[0])

    tee.write("=== Bottom Itineraries ===\n")
    for score, r in scored[:k]:
        tee.write(f"- {r['qid']} (overall={score:.1f})\n")
    tee.write("\n")

    tee.write("=== Top Itineraries ===\n")
    for score, r in scored[-k:]:
        tee.write(f"- {r['qid']} (overall={score:.1f})\n")
    tee.write("\n")


# ---------------------------
# Suspicious cases
# ---------------------------
def find_suspicious(results, min_hr, bad_feas_threshold, tee: Tee):
    tee.write(
        f"=== Suspicious Cases (HR >= {min_hr}, low feasibility <= {bad_feas_threshold}) ===\n"
    )

    susp = []
    for r in results:
        hr = safe_get(r, "HR", 0)
        if hr < min_hr:
            continue

        flags = []
        for k in ["route_feasibility_score", "poi_feasibility_score",
                  "semantic_feasibility_score", "overall_feasibility_score"]:
            val = safe_get(r, k)
            if val is not None and val <= bad_feas_threshold:
                flags.append(f"{k}={val:.1f}")

        if flags:
            susp.append((hr, r, ", ".join(flags)))

    if not susp:
        tee.write("- None\n\n")
        return

    # sort: highest HR, then lowest overall
    susp.sort(key=lambda x: (-x[0], safe_get(x[1], "overall_feasibility_score", 0)))

    for hr, r, flags in susp:
        tee.write(f"- {r['qid']} (HR={hr:.2f}) {flags}\n")
    tee.write("\n")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="evaluate_result")
    parser.add_argument("--output-file", type=str, default="evaluation_summary.txt")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-hr", type=float, default=0.9)
    parser.add_argument("--bad-feas-threshold", type=float, default=5.0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_file = Path(args.output_file)

    tee = Tee(out_file)

    results = load_results(results_dir)
    summarize(results, tee)
    print_top_bottom(results, args.top_k, tee)
    find_suspicious(results, args.min_hr, args.bad_feas_threshold, tee)

    tee.write(f"Saved summary to: {out_file}\n")
    tee.close()


if __name__ == "__main__":
    main()
