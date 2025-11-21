#!/usr/bin/env python3
"""
POI extraction + embedding pipeline from Reddit Japan travel posts.

- Reads:  travel_japan.jsonl
- Step 1: LLM → POI mentions + 4 dimensions
- Step 2: Embeddings → poi_mentions_with_emb.jsonl
"""

import json
import json_repair
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import jsonlines
from tqdm import tqdm
from openai import OpenAI


# ========================
# Config
# ========================

@dataclass
class Config:
    # Adjust paths if needed
    reddit_posts_path: Path = Path("travel_japan.jsonl")
    poi_mentions_path: Path = Path("poi_mentions.jsonl")
    poi_mentions_emb_path: Path = Path("poi_mentions_with_emb.jsonl")

    llm_model: str = "gpt-5-nano"          # change if needed
    embed_model: str = "text-embedding-3-large"


# ========================
# Data model
# ========================

@dataclass
class RedditPost:
    post_id: str
    title: str
    body: str

    @property
    def full_text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


# ========================
# Prompt (improved IER)
# ========================

IER_PROMPT = """
You are a travel analysis expert specializing in Japan. Your task is to read a Reddit travel post
and extract all **POIs (places)** the traveler visited or strongly recommends. For each POI,
extract four experiential dimensions based on the content of the post.

### Dimensions to Extract (per POI)

1. **Landscape and Content**  
   Describe the natural or built environment of this place as implied by the post.  
   Examples: shrine, park, street area, temple, shopping district, viewpoint, nightlife alley, museum, mountain, garden.

2. **Activities**  
   Describe what the traveler did (or typically does) at this POI.  
   Examples: eating, sightseeing, walking, shopping, relaxing, nightlife, taking photos, hiking.

3. **Atmosphere**  
   Describe the emotional or sensory tone of the place.  
   Examples: peaceful, busy, lively, romantic, traditional, modern, local, touristy, vibrant.

4. **Time and Schedule**  
   Extract any temporal clues about when or how the POI was visited.  
   Examples: "morning", "at night", "sunset", "quick stop", "half-day", "cherry blossom season".  
   If not provided, return "unknown".

---

### Required Output Format (strict JSON list)

Return **only** a JSON list.  
Each item corresponds to one POI and contains **exactly** these keys:

{
  "city": "...",
  "poi_name": "...",
  "poi_type": "...",
  "landscape and content": "...",
  "activities": "...",
  "atmosphere": "...",
  "time and schedule": "..."
}

- If the information is missing, infer lightly or use "unknown".  
- Do **not** include explanations or text outside the JSON.

---

### Reddit Post
{post_text}
"""


# ========================
# Core functions
# ========================

def load_reddit_posts(path: Path) -> List[RedditPost]:
    """Load posts from JSONL. Supports several common Reddit export fields."""
    posts: List[RedditPost] = []
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            post_id = str(obj.get("id") or obj.get("post_id"))
            title = obj.get("title", "") or ""
            body = (
                obj.get("text")
                or obj.get("body")
                or obj.get("selftext")
                or ""
            )
            posts.append(RedditPost(post_id=post_id, title=title, body=body))
    return posts


def extract_poi_mentions_for_post(
    client: OpenAI, cfg: Config, post: RedditPost
) -> List[dict]:
    """
    Call LLM with the improved IER prompt and return list[dict] of POI mentions.
    """
    prompt = IER_PROMPT.replace("{post_text}", post.full_text)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict JSON generator. "
                "Return ONLY a valid JSON list following the required schema, "
                "with no explanations or extra text."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for attempt in range(cfg.max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.llm_model,
                messages=messages,
                temperature=cfg.temperature,
            )
            raw = resp.choices[0].message.content or ""
            content = raw.strip()

            # Strip Markdown code fences if present
            if content.startswith("```"):
                lines = content.splitlines()
                if lines and lines[0].lstrip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].lstrip().startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            # Try normal JSON first, then fall back to json_repair
            try:
                data = json.loads(content)
            except Exception:
                try:
                    data = json_repair.loads(content)
                except Exception:
                    print(f"[WARN] invalid JSON for post {post.post_id}: {raw[:300]!r}")
                    raise

            # Normalize to a list
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                print(f"[WARN] non-list JSON root for post {post.post_id}: {type(data)}")
                return []

            # Keep only dicts and strip whitespace from keys
            cleaned: List[dict] = []
            for item in data:
                if isinstance(item, dict):
                    fixed = {}
                    for k, v in item.items():
                        if isinstance(k, str):
                            fixed[k.strip()] = v
                    cleaned.append(fixed)

            return cleaned
        except Exception as e:
            print(f"[WARN] post {post.post_id} attempt {attempt+1} failed: {e}")
    return []  # on failure, skip this post


def run_extraction(cfg: Config):
    """Step 1: Reddit posts -> structured POI mentions (no embeddings)."""
    client = OpenAI()
    posts = load_reddit_posts(cfg.reddit_posts_path)

    cfg.poi_mentions_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(cfg.poi_mentions_path, "w") as writer:
        iterable = posts[:cfg.limit] if getattr(cfg, "limit", None) else posts
        for post in tqdm(iterable, desc=f"Extracting POIs (limit={cfg.limit})"):
            poi_dicts = extract_poi_mentions_for_post(client, cfg, post)
            for i, d in enumerate(poi_dicts):
                if not isinstance(d, dict):
                    print(f"[WARN] Skipping non-dict item in post {post.post_id}: {d!r}")
                    continue
                writer.write(
                    {
                        "mention_id": f"{post.post_id}_{i}",
                        "post_id": post.post_id,
                        "city": d.get("city", ""),
                        "poi_name": d.get("poi_name", ""),
                        "poi_type": d.get("poi_type", ""),
                        "landscape_content": d.get("landscape and content", ""),
                        "activities": d.get("activities", ""),
                        "atmosphere": d.get("atmosphere", ""),
                        "time_schedule": d.get("time and schedule", ""),
                    }
                )


def build_embedding_text(poi: dict) -> str:
    """Create a single text string for embedding from the POI + dimensions."""
    return (
        f"{poi['poi_name']} ({poi['poi_type']}) in {poi['city']}. "
        f"Landscape: {poi.get('landscape_content', '')}. "
        f"Activities: {poi.get('activities', '')}. "
        f"Atmosphere: {poi.get('atmosphere', '')}. "
        f"Time: {poi.get('time_schedule', '')}."
    )


def run_embeddings(cfg: Config):
    """Step 2: add embeddings to each POI mention."""
    client = OpenAI()

    with jsonlines.open(cfg.poi_mentions_path, "r") as reader, \
         jsonlines.open(cfg.poi_mentions_emb_path, "w") as writer:

        for poi in tqdm(reader, desc="Embedding POIs"):
            text = build_embedding_text(poi)
            try:
                emb_resp = client.embeddings.create(
                    model=cfg.embed_model,
                    input=text,
                )
                poi["embedding"] = emb_resp.data[0].embedding
            except Exception as e:
                print(f"[WARN] embedding failed for {poi.get('mention_id')}: {e}")
                poi["embedding"] = None
            writer.write(poi)


# ========================
# CLI
# ========================

def main():
    parser = argparse.ArgumentParser(
        description="POI extraction + embedding pipeline from Reddit Japan posts"
    )
    parser.add_argument(
        "--step",
        choices=["extract", "embed", "all"],
        default="all",
        help="Which step to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of posts to process (e.g., 30, 50, 100)."
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.limit = args.limit

    if args.step in ("extract", "all"):
        run_extraction(cfg)

    if args.step in ("embed", "all"):
        run_embeddings(cfg)


if __name__ == "__main__":
    main()