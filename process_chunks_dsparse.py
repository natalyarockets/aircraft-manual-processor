#!/usr/bin/env python3
"""
process_chunks_dsparse.py
Minimal wrapper for dsparse that:
- Reads PDF path from config.PDF_PATH (no CLI file arg required)
- Loads API keys from .env via python-dotenv (no exports)
- Optionally skips front matter heuristically
- Prints a few chunks to stdout, and can write raw chunks JSON

Usage:
  python process_chunks_dsparse.py --show 8 --filter-front-matter --out chunks.json

Requires:
  pip install dsparse python-dotenv
  # And poppler for PDF→image:
  # macOS:  brew install poppler
  # Debian: sudo apt-get update && sudo apt-get install -y poppler-utils
"""

from __future__ import annotations
import os, sys, json, argparse, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()  # load GOOGLE_API_KEY, etc. from .env

# Your config with the PDF path
try:
    from config import PDF_PATH
except Exception as e:
    print("ERROR: Failed to import PDF_PATH from config.py", file=sys.stderr)
    raise

# dsparse library
try:
    #from dsparse import parse_and_chunk
    from dsparse.main import parse_and_chunk
except Exception:
    print("ERROR: dsparse is not installed. Run: pip install dsparse python-dotenv", file=sys.stderr)
    raise

# Defaults from env (override via env only)
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "gemini")
VLM_MODEL    = os.getenv("VLM_MODEL", "gemini-2.0-flash")
EXCLUDE      = os.getenv("DSPARSE_EXCLUDE", "Header,Footer")

def ensure_auth(provider: str) -> None:
    p = provider.lower()
    if p == "gemini":
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY). Put it in your .env.")
    elif p == "vertex":
        # Expect ADC via GOOGLE_APPLICATION_CREDENTIALS (service account), no API key.
        pass
    else:
        raise ValueError(f"Unknown provider: {provider}")

# --- Simple front-matter filter (optional) ---
FRONT_MATTER_PATTERNS = re.compile(
    r"(table of contents|list of effective pages|record of revisions|revision record|"
    r"log of revisions|copyright|proprietary notice|distribution statement|toc\b|lep\b)",
    re.IGNORECASE,
)

def looks_like_front_matter(chunk: Dict[str, Any]) -> bool:
    text = (chunk.get("content") or chunk.get("text") or "").strip()
    if not text:
        return True
    if FRONT_MATTER_PATTERNS.search(text):
        return True
    if len(text) < 200 and re.search(r"(copyright|revision|toc|lep)", text, re.I):
        return True
    return False

def first_content_index(chunks: List[Dict[str, Any]]) -> int:
    for i, c in enumerate(chunks):
        txt = (c.get("content") or c.get("text") or "").lower()
        if not looks_like_front_matter(c):
            if any(k in txt for k in ("introduction", "general description", "system description", "normal procedures")):
                return i
            if len(txt) > 400:
                return i
    return 0

def main():
    ap = argparse.ArgumentParser(description="Parse & chunk PDF from config.PDF_PATH with dsparse and print sample chunks.")
    ap.add_argument("--show", type=int, default=6, help="How many chunks to print")
    ap.add_argument("--out", default=None, help="Optional path to write raw chunks JSON")
    ap.add_argument("--filter-front-matter", action="store_true", help="Heuristically skip front matter before showing chunks")
    args = ap.parse_args()

    pdf_path = str(Path(PDF_PATH).expanduser())
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    ensure_auth(VLM_PROVIDER)

    exclude_elements = [e.strip() for e in EXCLUDE.split(",") if e.strip()]
    file_parsing_config = {
        "use_vlm": True,
        "exclude_elements": exclude_elements,
        "vlm_config": {
            "provider": VLM_PROVIDER,
            "model": VLM_MODEL,
            # For Vertex (if you switch): include project/location via env if dsparse supports it
            # "project": os.getenv("GOOGLE_CLOUD_PROJECT"),
            # "location": os.getenv("GOOGLE_LOCATION", "us-central1"),
        },
    }

    print(f"➡️  dsparse.parse_and_chunk(file={pdf_path}) | provider={VLM_PROVIDER} model={VLM_MODEL}")
    sections, chunks = parse_and_chunk(
        kb_id="tmp",
        doc_id=Path(pdf_path).stem,
        file_path=pdf_path,
        file_parsing_config=file_parsing_config,
    )
    print(f"   Sections: {len(sections)} | Chunks: {len(chunks)}")

    view = chunks
    start_idx = 0
    if args.filter_front_matter and chunks:
        start_idx = first_content_index(chunks)
        view = chunks[start_idx:]
        first_page = view[0].get("page_number") if view else "?"
        print(f"   Front-matter skip: starting at chunk index {start_idx} (page ~{first_page})")

    print("\n=== SAMPLE CHUNKS ===")
    for i, ch in enumerate(view[: args.show], start=0):
        txt = (ch.get("content") or ch.get("text") or "").strip()
        page = ch.get("page_number") or ch.get("page") or ch.get("page_index")
        title = ch.get("section_title") or ch.get("title") or ch.get("section")
        etype = ch.get("element_type") or ch.get("type")
        preview = re.sub(r"\s+", " ", txt)[:800]
        print(f"\n[{i+start_idx}] page={page} title={title!r} type={etype!r}")
        print(preview if preview else "(empty)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Wrote raw chunks JSON → {out_path}")

if __name__ == "__main__":
    main()
