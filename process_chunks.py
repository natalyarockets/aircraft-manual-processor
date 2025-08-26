import os
import re
import fitz  # PyMuPDF
import json
import tiktoken
from typing import List, Dict, Tuple, Optional
from config import AIRCRAFT_MODEL, PDF_PATH, RAW_CHUNKS_PATH
from collections import defaultdict


# Optional: enable if/when you want to add LLM cleanup
USE_LLM_CLEANUP = True

# Tokenizer for OpenAI-style models
tokenizer = tiktoken.get_encoding("cl100k_base")

# --- categorization config ---
_CAT_PATTERNS = {
    "Non-Content": [
        (re.compile(r"\b(table of contents|contents|toc|list of (tables|figures)|illustrations?|drawings?|schematics?)\b", re.I), 3.0),
    ],
    "Hydraulic": [
        (re.compile(r"\bhydraul\w+\b", re.I), 3.0),
        (re.compile(r"\b(actuator|reservoir|accumulator|servo|pressure|relief valve|hyd\.? pump)\b", re.I), 1.5),
    ],
    "Electrical": [
        (re.compile(r"\belectrical?( system)?\b", re.I), 3.0),
        (re.compile(r"\b(circuit(s)?|voltage|current|battery|generator|alternator|bus(bar)?|breaker|inverter)\b", re.I), 1.5),
    ],
    "Avionics": [
        (re.compile(r"\b(avionics?|navigation|communication|radar|(fms|gps)\b|display|computer)\b", re.I), 1.5),
        (re.compile(r"\bflight director\b", re.I), 3.0),
    ],
    "Powerplant": [
        (re.compile(r"\b(powerplant|engine(s)?|turbine|compressor|combustion|exhaust|ignition|propeller)\b", re.I), 2.0),
        (re.compile(r"\b(n1|n2|egt|itt|torque)\b", re.I), 1.2),
    ],
    "Landing Gear": [
        (re.compile(r"\blanding gear\b", re.I), 3.0),
        (re.compile(r"\b(strut|shock|brake(s)?|wheel(s)?|tire(s)?|nose gear|extend|retract)\b", re.I), 1.5),
    ],
    "Maintenance": [
        (re.compile(r"\b(troubleshooting|inspection|intervals?|schedule|service|lubricat(e|ion)|maintenance)\b", re.I), 1.5),
        (re.compile(r"\b(removal and installation|R&I|install(ation)?|remove)\b", re.I), 1.5),
    ],
    "Fuel System": [
        (re.compile(r"\bfuel system\b", re.I), 3.0),
        (re.compile(r"\b(fuel (tank|selector|quantity|indicator|flow)|boost pump|unfeather pump)\b", re.I), 1.8),
    ],
    "Environmental": [
        (re.compile(r"\b(air conditioning|pressurization|bleed air|ventilation|temperature control|oxygen)\b", re.I), 2.0),
    ],
    "Flight Controls": [
        (re.compile(r"\b(flight control|aileron|elevator|rudder|flap|spoiler|trim)\b", re.I), 1.8),
        (re.compile(r"\b(autopilot|yaw damper)\b", re.I), 2.2),
    ],
    "Emergency Procedures": [
        (re.compile(r"\b(emergency procedures?|abnormal|warning(s)?|caution(s)?)\b", re.I), 1.5),
    ],
    "Limitations": [
        (re.compile(r"\b(limitations?|operating limits?|weight and balance|center of gravity|c\.?g\.?)\b", re.I), 1.8),
    ],
    "Operating Procedures": [
        (re.compile(r"\b(normal procedures?|operational procedures?|checklist|preflight|postflight|start(ing)?|shutdown)\b", re.I), 1.5),
    ],
}

_TITLE_WEIGHT = 3.0
_TEXT_WEIGHT  = 1.0
_MIN_SCORE    = 2.0


def detect_system_category(text: str, title: Optional[str] = None, topk: int = 1):
    """
    Count weighted keyword/phrase hits across categories in both title and body.
    Returns the top category (or top-k list) based on scores.
    """
    scores = defaultdict(float)
    body = text or ""
    hdr  = title or ""

    for cat, pats in _CAT_PATTERNS.items():
        for pat, w in pats:
            if hdr:
                scores[cat] += _TITLE_WEIGHT * w * len(pat.findall(hdr))
            scores[cat] += _TEXT_WEIGHT * w * len(pat.findall(body))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] < _MIN_SCORE:
        return "General"

    if topk == 1:
        return ranked[0][0]
    return [c for c, _ in ranked[:topk]]


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    print(f"Extracting text from: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text and len(text.strip()) > 50:
            pages.append({
                "page_number": i + 1,
                "text": text.strip()
            })
        elif i % 50 == 0:
            print(f"Warning: Page {i+1} in {os.path.basename(pdf_path)} may have minimal or no extractable content")
    print(f"Extracted {len(pages)} pages of usable content from {os.path.basename(pdf_path)}")
    return pages

def llm_cleanup(text: str) -> str:
    cleaned = text.replace("\x00", "")
    #*Could do other things here like removing excessive whitespace, etc.
    return cleaned

def chunk_text(pages_text: List[Dict], pdf_basename: str, chunk_size: int = 600, chunk_overlap: int = 50) -> List[Dict]:
    print(f"Chunking text for: {pdf_basename}")
    chunks = []
    for page in pages_text:
        page_number = page["page_number"]
        # Normalize whitespace
        text = re.sub(r'\s{2,}', ' ', page["text"]).strip()
        tokens = tokenizer.encode(text)

        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)

            if USE_LLM_CLEANUP:
                chunk_text = llm_cleanup(chunk_text)

            # Heuristic section title
            section_title = "Unknown Section"
            title_match = re.search(r'(?:^|\n)(?:Chapter|Section|Part|Item|Task)?\s*[\d\.\-]*\s*([A-Z][^\n\.]{3,60})', chunk_text)
            if title_match:
                section_title = title_match.group(1).strip()

            system_category = detect_system_category(chunk_text, title=section_title, topk=1)

            chunks.append({
                "content": chunk_text,
                "page_number": page_number,
                "section_title": section_title,
                "aircraft_model": AIRCRAFT_MODEL,
                "document_id": pdf_basename,  # << per-file identifier
                "system_category": system_category,
                "token_count": len(chunk_tokens),
            })
    print(f"Created {len(chunks)} chunks for {pdf_basename}")
    return chunks

def list_pdf_files(path: str) -> List[str]:
    if os.path.isdir(path):
        pdfs = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".pdf")]
        return sorted(pdfs)
    elif os.path.isfile(path) and path.lower().endswith(".pdf"):
        return [path]
    else:
        raise FileNotFoundError(f"No PDF or directory found at: {path}")

def save_chunks(chunks: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Chunks saved to {output_path}")

def main():
    print(f"Preparing chunks for: {PDF_PATH}")
    pdf_files = list_pdf_files(PDF_PATH)

    # If RAW_CHUNKS_PATH ends with .json => combine all into one file
    combine = RAW_CHUNKS_PATH.lower().endswith(".json")

    all_chunks: List[Dict] = []
    for pdf in pdf_files:
        basename = os.path.basename(pdf)
        pages = extract_text_from_pdf(pdf)
        if not pages:
            print(f"No text found in {basename}. Skipping.")
            continue

        chunks = chunk_text(pages, pdf_basename=basename)
        if not chunks:
            print(f"No chunks created for {basename}. Skipping.")
            continue

        if combine:
            all_chunks.extend(chunks)
        else:
            # Treat RAW_CHUNKS_PATH as a folder; write per-PDF JSON
            out_dir = RAW_CHUNKS_PATH
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{os.path.splitext(basename)[0]}.chunks.json")
            save_chunks(chunks, out_file)

    if combine:
        # One combined output file
        save_chunks(all_chunks, RAW_CHUNKS_PATH)

if __name__ == "__main__":
    main()
