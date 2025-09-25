#!/usr/bin/env python3
"""
Build a canonical list of common squawks from real data and pre-populate research context.

This script:
  - Reads squawks (text + embedding) from `squawk_text_embedding_3_small_demo` (READ-ONLY).
  - Clusters embeddings (HDBSCAN if available; KMeans fallback).
  - Synthesizes a canonical/common squawk per cluster (LLM or heuristic fallback).
  - Ranks clusters by frequency and keeps top N.
  - (Stub) Gathers research context from other Supabase projects (manuals/overviews/past).
  - UPSERTS rows into `common_squawks`.

It does NOT modify the demo table.

Environment:
  SUPABASE_URL_PRIMARY, SUPABASE_KEY_PRIMARY         -> where the demo table and common_squawks live (or split if you prefer)
  SUPABASE_URL_MANUALS, SUPABASE_KEY_MANUALS         -> placeholder project for manuals
  SUPABASE_URL_OVERVIEWS, SUPABASE_KEY_OVERVIEWS     -> placeholder project for overviews
  SUPABASE_URL_PAST, SUPABASE_KEY_PAST               -> placeholder project for past squawks
  OPENAI_API_KEY
"""

import os
import math
import json
import logging
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---- Clustering deps
USE_HDBSCAN = True
try:
    import hdbscan  # pip install hdbscan
except Exception:
    USE_HDBSCAN = False

import numpy as np
from sklearn.cluster import KMeans

# ---- Supabase + OpenAI
from supabase import create_client, Client
from openai import OpenAI  # pip install openai>=1.0.0

# ---------------------------
# Config
# ---------------------------
PRIMARY_SCHEMA = "public"
PRIMARY_TABLE = "squawk_text_embedding_3_small_demo"  # source (read-only here)
COMMON_TABLE  = "common_squawks"                       # destination (research)

TOP_N_COMMON = 200        # keep top-N clusters by frequency
EXAMPLES_PER_COMMON = 5   # store a few examples per canonical
MAX_ROWS = None           # cap for development, e.g. 20000
KMEANS_CLUSTERS = 300     # fallback KMeans cluster count
OPENAI_MODEL = "gpt-4o-mini"

# ---------------------------
# Clients
# ---------------------------
def make_supabase(url_env: str, key_env: str) -> Client:
    url = os.getenv(url_env, "https://YOUR-PROJECT.supabase.co")
    key = os.getenv(key_env, "YOUR_SERVICE_ROLE_OR_ANON_KEY")
    return create_client(url, key)

sb_primary  = make_supabase("SUPABASE_URL_PRIMARY",  "SUPABASE_KEY_PRIMARY")
sb_manuals  = make_supabase("SUPABASE_URL_MANUALS",  "SUPABASE_KEY_MANUALS")
sb_overview = make_supabase("SUPABASE_URL_OVERVIEWS","SUPABASE_KEY_OVERVIEWS")
sb_past     = make_supabase("SUPABASE_URL_PAST",     "SUPABASE_KEY_PAST")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ---------------------------
# Data access
# ---------------------------
def fetch_squawks(limit: int | None = MAX_ROWS) -> List[Dict[str, Any]]:
    logging.info("Fetching squawks from primary...")
    q = sb_primary.table(PRIMARY_TABLE).select("id,text,embedding")
    if limit:
        q = q.limit(limit)
    rows = q.execute().data
    rows = [r for r in rows if r.get("embedding") and r.get("text")]
    logging.info(f"Fetched {len(rows)} rows with embeddings.")
    return rows

# ---------------------------
# Clustering
# ---------------------------
def cluster_embeddings(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([np.array(r["embedding"], dtype=np.float32) for r in rows])

    if USE_HDBSCAN:
        logging.info("Clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean")
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logging.info(f"HDBSCAN produced {n_clusters} clusters (+ noise).")
        return labels, X

    logging.info("Clustering with KMeans fallback...")
    k = min(KMEANS_CLUSTERS, max(2, (len(X) // 30) or 10))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    logging.info(f"KMeans produced {len(set(labels))} clusters.")
    return labels, X

# ---------------------------
# Canonicalization
# ---------------------------
CANON_PROMPT = """You are normalizing aircraft maintenance "squawks" (pilot-reported issues).
Given EXAMPLES of very similar squawks, produce ONE short, generalized description (max ~8 words) that captures the common issue pattern without aircraft-specific tail numbers or extraneous detail.
Return just the phrase.

EXAMPLES:
{s}
"""

def canonicalize_squawk(examples: List[str]) -> str:
    examples = [e.strip() for e in examples if isinstance(e, str) and e.strip()]
    examples = examples[:10] or ["Noisy placeholder"]
    if not os.getenv("OPENAI_API_KEY"):
        # Simple heuristic fallback: shortest example, normalized
        s = min(examples, key=len).lower()
        return " ".join(s.split())[:120]

    msg = CANON_PROMPT.format(s="\n- " + "\n- ".join(examples))
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Be concise and precise."},
            {"role": "user", "content": msg},
        ],
        temperature=0.1,
        max_tokens=32,
    )
    text = resp.choices[0].message.content.strip()
    return " ".join(text.split())[:120]

# ---------------------------
# Research context (stubs)
# ---------------------------
def fetch_research_context(canonical: str) -> Dict[str, Any]:
    """
    Wire these to your cross-project tables later, e.g.:
      - sb_manuals.table("maintenance_manual_chunks").select(...).text_search("content", canonical).limit(5)
      - sb_overview.table("aircraft_overviews").select(...).text_search("content", canonical).limit(3)
      - sb_past.table("past_squawks").select(...).text_search("text", canonical).limit(5)
    """
    manuals, overviews, past = [], [], []
    return {"manuals": manuals, "overviews": overviews, "past_squawks": past}

# ---------------------------
# Upserts into research table ONLY
# ---------------------------
def upsert_common_squawks(payload: List[Dict[str, Any]]) -> None:
    """
    Payload items:
      {
        "canonical_squawk": str,
        "cluster_label": int,
        "frequency": int,
        "sample_examples": List[str],
        "sources": Dict
      }
    Uses unique index on (canonical_squawk) for stable upserts.
    """
    if not payload:
        return
    # Batch to avoid large bodies
    BATCH = 500
    for i in range(0, len(payload), BATCH):
        chunk = payload[i:i+BATCH]
        sb_primary.table(COMMON_TABLE).upsert(
            chunk,
            on_conflict="canonical_squawk",  # requires unique index created in SQL
            ignore_duplicates=False
        ).execute()

# ---------------------------
# Main
# ---------------------------
def main():
    rows = fetch_squawks(MAX_ROWS)
    if not rows:
        logging.warning("No rows to process.")
        return

    labels, X = cluster_embeddings(rows)

    # Group by label (skip HDBSCAN noise = -1)
    clusters: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        if lbl == -1:  # treat noise as its own tiny clusters? For now, skip.
            continue
        clusters.setdefault(lbl, []).append(idx)

    # Rank clusters by frequency
    ranked = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    top = ranked[:TOP_N_COMMON]

    # Build payload for upsert
    out: List[Dict[str, Any]] = []
    for lbl, idxs in top:
        examples = [rows[i]["text"] for i in idxs]
        canonical = canonicalize_squawk(examples)
        sources = fetch_research_context(canonical)
        out.append({
            "canonical_squawk": canonical,
            "cluster_label": int(lbl),
            "frequency": len(idxs),
            "sample_examples": examples[:EXAMPLES_PER_COMMON],
            "sources": sources,
        })

    logging.info(f"Upserting {len(out)} common squawks into '{COMMON_TABLE}'...")
    upsert_common_squawks(out)
    logging.info("Done. (Demo table was not modified.)")

if __name__ == "__main__":
    main()
