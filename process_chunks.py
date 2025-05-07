import os
import re
import fitz  # PyMuPDF
import json
import tiktoken
from config import AIRCRAFT_MODEL, PDF_PATH, RAW_CHUNKS_PATH

# Optional: enable if/when you want to add LLM cleanup
USE_LLM_CLEANUP = False

# Tokenizer for OpenAI-style models
tokenizer = tiktoken.get_encoding("cl100k_base")

def detect_system_category(text):
    text_lower = text.lower()
    if re.search(r'hydraulic|actuator|fluid|pressure|reservoir|pump|valve|cylinder', text_lower):
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

def extract_text_from_pdf(pdf_path):
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
            print(f"Warning: Page {i+1} may have minimal or no extractable content")
    print(f"Extracted {len(pages)} pages of usable content")
    return pages

def chunk_text(pages_text, chunk_size=600, chunk_overlap=50):
    print("Chunking text...")
    chunks = []
    for page in pages_text:
        page_number = page["page_number"]
        text = re.sub(r'\s{2,}', ' ', page["text"]).strip()
        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)

            # Optional: LLM cleanup (off by default)
            if USE_LLM_CLEANUP:
                chunk_text = llm_cleanup(chunk_text)

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
                "document_id": os.path.basename(PDF_PATH),
                "system_category": system_category,
                "token_count": len(chunk_tokens)
            })
    print(f"Created {len(chunks)} chunks")
    return chunks

# Optional cleanup stub – replace with actual call to OpenAI/GPT if needed
def llm_cleanup(text):
    return text  # stub for now

def save_chunks(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Chunks saved to {output_path}")

def main():
    print(f"Preparing chunks for: {PDF_PATH}")
    pages = extract_text_from_pdf(PDF_PATH)
    if not pages:
        print("No text found. Exiting.")
        return

    chunks = chunk_text(pages)
    if not chunks:
        print("No chunks created. Exiting.")
        return

    save_chunks(chunks, RAW_CHUNKS_PATH)

if __name__ == "__main__":
    main()
