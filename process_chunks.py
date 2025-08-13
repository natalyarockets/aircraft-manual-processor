import os
import re
import fitz  # PyMuPDF
import json
import tiktoken
from typing import List, Dict
from config import AIRCRAFT_MODEL, PDF_PATH, RAW_CHUNKS_PATH

# Optional: enable if/when you want to add LLM cleanup
USE_LLM_CLEANUP = True

# Tokenizer for OpenAI-style models
tokenizer = tiktoken.get_encoding("cl100k_base")

def detect_system_category(text: str) -> str:
    text_lower = text.lower()
    if re.search(r'(table of contents|contents|toc\b|list of tables|list of figures|illustrations|drawings|schematic|wiring diagram)', text_lower):
        return "Non-Content"
    elif re.search(r'hydraulic|actuator|fluid|pressure|reservoir|pump|valve|cylinder', text_lower):
        return "Hydraulic"
    elif re.search(r'electric|circuit|voltage|wire|battery|generator|power|bus|breaker', text_lower):
        return "Electrical"
    elif re.search(r'avionic|instrument|display|navigation|communication|radar|computer|software', text_lower):
        return "Avionics"
    elif re.search(r'engine|thrust|power|fuel|combustion|turbine|compressor|exhaust|ignition', text_lower):
        return "Powerplant"
    elif re.search(r'landing|gear|wheel|brake|tire|strut|shock|nose gear|extend|retract', text_lower):
        return "Landing Gear"
    elif re.search(r'inspect|check|maintenance|interval|schedule|service|lubricate|procedure', text_lower):
        return "Maintenance"
    elif re.search(r'fuel|tank|pump|valve|selector|quantity|indicator|flow|consumption', text_lower):
        return "Fuel System"
    elif re.search(r'environmental|air|conditioning|pressurization|temperature|ventilation|oxygen', text_lower):
        return "Environmental"
    elif re.search(r'flight control|aileron|elevator|rudder|flap|spoiler|trim|autopilot', text_lower):
        return "Flight Controls"
    elif re.search(r'emergency|procedure|warning|caution|danger|abnormal|failure|malfunction', text_lower):
        return "Emergency Procedures"
    elif re.search(r'limitation|performance|weight|balance|loading|center of gravity|cg', text_lower):
        return "Limitations"
    elif re.search(r'operational|normal|procedure|checklist|preflight|postflight', text_lower):
        return "Operating Procedures"
    return "General"

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

            system_category = detect_system_category(chunk_text)

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
