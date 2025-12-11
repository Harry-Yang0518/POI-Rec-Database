# POI-Rec-Database  
End-to-end pipeline for POI retrieval, multi-day itinerary planning, and LLM-based evaluation.

---

## 📐 Pipeline Overview

The system consists of six modular stages:

```text
0_gen_user_query.py      Generate structured user queries
1_top_k.py               Retrieve top-k POIs via embedding similarity
2_plan_route.py          Plan multi-day itineraries (city grouping, distance ordering)
3_evaluate.py            LLM-based itinerary evaluation
4_run_full_pipeline.py   Orchestrate full pipeline (Stage 0 → 3)
5_evaluate_all.py        Batch evaluation across all itineraries
```

---

## 📁 Directory Structure

```text
data/                 # POI cards, raw JSONL, embeddings
topk_pick/            # Retrieved top-k POIs
itinerary/            # Generated multi-day itineraries
evaluate_result/      # LLM evaluation outputs
_cache/               # Local embedding cache
misc/                 # Extra assets/utilities
```

---

## 🧩 Stage 0 — User Query Generation

```bash
python 0_gen_user_query.py \
  --output data/user_queries_japan.jsonl
```

Generates structured requests (days, start/end city, budget style, season, etc.).

---

## 🔍 Stage 1 — Top-k Retrieval

```bash
python 1_top_k.py \
  --user-queries data/user_queries_japan.jsonl \
  --poi-db data/poi_cards_structured.json
```

Embeds POIs and retrieves relevant candidates per query.

---

## 🗺️ Stage 2 — Multi-Day Itinerary Planning

```bash
python 2_plan_route.py \
  --qid q00002 \
  --topk-path topk_pick/topk_q00002.json \
  --poi-db data/poi_cards_structured.json
```

Performs city assignment, distance-aware ordering, and time-of-day schedule generation.

---

## 📝 Stage 3 — Itinerary Evaluation

```bash
python 3_evaluate.py \
  --qid q00002 \
  --user-queries-path data/user_queries_japan.jsonl
```

LLM judges provide Hit Rate (HR), POI/route feasibility, semantic consistency, and overall scores.

---

## 🚀 Full Pipeline Execution

```bash
python 4_run_full_pipeline.py \
  --start 0 --end 20 \
  --user-queries data/user_queries_japan.jsonl
```

Runs user query → retrieval → routing → evaluation end-to-end.

---

## 📊 Batch Evaluation

```bash
python 5_evaluate_all.py \
  --results-dir evaluate_result/
```

Aggregates results for all itineraries.

---

## 📦 Install Requirements

```bash
pip install -r requirements.txt
```

---

## 📄 Citation
If you use this pipeline, please cite the corresponding report or repository.
