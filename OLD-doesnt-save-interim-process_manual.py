import os
import numpy as np
import pypdf
import re
import tiktoken
import time
from openai import OpenAI
from supabase import create_client, Client
from config import *

# Initialize OpenAI and Supabase clients
#SUPABASE_VS_URL = os.environ.get("SUPABASE_VS_URL")
#SUPABASE_VS_KEY = os.environ.get("SUPABASE_VS_KEY")
#OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_VS_URL, SUPABASE_VS_KEY)

# Initialize tokenizer for counting tokens
tokenizer = tiktoken.get_encoding("cl100k_base")  # This works for OpenAI embedding models

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file with enhanced error handling for aircraft manuals"""
    print(f"Extracting text from {pdf_path}...")
    
    try:
        pdf_reader = pypdf.PdfReader(pdf_path)
        num_pages = len(pdf_reader.pages)
        
        pages_text = []
        problematic_pages = []
        
        for i in range(num_pages):
            if i % 20 == 0:
                print(f"Processing page {i+1} of {num_pages}...")
            
            try:
                page = pdf_reader.pages[i]
                text = page.extract_text()
                
                # Check if page has meaningful content
                if text and len(text.strip()) > 50:  # Require at least 50 chars
                    pages_text.append({
                        "page_number": i + 1,
                        "text": text
                    })
                else:
                    problematic_pages.append(i + 1)
                    if i % 100 == 0:  # Don't log too frequently
                        print(f"Warning: Page {i+1} has minimal text content (potentially a diagram page)")
            
            except Exception as e:
                problematic_pages.append(i + 1)
                print(f"Error extracting page {i+1}: {str(e)}")
        
        print(f"Extracted text from {len(pages_text)} pages successfully")
        if problematic_pages:
            print(f"Found {len(problematic_pages)} problematic pages that might need manual review")
            if len(problematic_pages) < 50:  # Only show if the list isn't too long
                print(f"Problematic pages: {problematic_pages}")
        
        return pages_text
    
    except Exception as e:
        print(f"Critical error extracting PDF: {str(e)}")
        return []

def chunk_text(pages_text, chunk_size=600, chunk_overlap=50):
    """Split text into chunks with page numbers - optimized for aircraft manuals"""
    print("Chunking text with size:", chunk_size, "and overlap:", chunk_overlap)
    chunks = []
    
    for page in pages_text:
        page_number = page["page_number"]
        text = page["text"]
        
        # Clean the text while preserving important whitespace structure
        text = re.sub(r'\s{2,}', ' ', text).strip()
        
        # Tokenize the text
        tokens = tokenizer.encode(text)
        
        # Split into chunks
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            # Enhanced pattern for aircraft manual section titles
            # Looking for patterns like "5.3.2 Landing Gear Inspection" or "Chapter 7 - Fuel System"
            section_title = "Unknown Section"
            title_match = re.search(r'(?:^|\n)(?:Chapter|Section|Part|Item|Task)?\s*[\d\.\-]*\s*([A-Z][^\n\.]{3,60})', chunk_text)
            if title_match:
                section_title = title_match.group(1).strip()
            
            # Detect system category based on content keywords
            system_category = detect_system_category(chunk_text)
            
            # Create chunk with enhanced metadata
            chunks.append({
                "content": chunk_text,
                "page_number": page_number,
                "section_title": section_title,
                "aircraft_model": AIRCRAFT_MODEL,
                "document_id": os.path.basename(PDF_PATH),
                "system_category": system_category
            })
    
    print(f"Created {len(chunks)} chunks")
    return chunks

def detect_system_category(text):
    """Detect the system category based on content keywords"""
    text_lower = text.lower()
    
    # Aircraft systems categorization based on keyword presence
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

def generate_embeddings(chunks):
    """Generate embeddings for chunks using OpenAI API with robust error handling"""
    print(f"Generating embeddings using {EMBEDDING_MODEL}...")
    total_chunks = len(chunks)
    
    # Track progress
    successful = 0
    failed = 0
    retry_attempts = 3
    
    for i, chunk in enumerate(chunks):
        if i % 20 == 0:
            print(f"Processing chunk {i+1} of {total_chunks}... ({successful} successful, {failed} failed)")
        
        # Try multiple times before giving up on a chunk
        for attempt in range(retry_attempts):
            try:
                response = openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=chunk["content"]
                )
                
                # Extract embedding from response
                embedding = response.data[0].embedding
                chunk["embedding"] = embedding
                successful += 1
                
                # Break out of retry loop on success
                break
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    # Not the last attempt, retry after a pause
                    print(f"Error on chunk {i+1}, attempt {attempt+1}: {str(e)}. Retrying...")
                    time.sleep(5)  # Longer pause before retry
                else:
                    # Last attempt failed
                    print(f"Failed to generate embedding for chunk {i+1} after {retry_attempts} attempts: {str(e)}")
                    chunk["embedding"] = None
                    failed += 1
        
        # Rate limiting to avoid OpenAI API limits - more conservative
        if (i + 1) % 50 == 0:
            pause_time = 5
            print(f"Processed {i + 1} chunks. Pausing for {pause_time}s to avoid rate limits...")
            time.sleep(pause_time)
    
    # Filter out chunks with failed embeddings
    valid_chunks = [c for c in chunks if c["embedding"] is not None]
    print(f"Embedding generation complete: {len(valid_chunks)} successful, {failed} failed")
    
    return valid_chunks

def upload_to_supabase(chunks):
    """Upload chunks with embeddings to Supabase with enhanced error handling"""
    print("Uploading to Supabase...")
    
    total_chunks = len(chunks)
    batch_size = 50  # Smaller batch size for more reliable uploads
    
    successful = 0
    failed = 0
    
    for i in range(0, total_chunks, batch_size):
        current_batch = i // batch_size + 1
        total_batches = (total_chunks - 1) // batch_size + 1
        print(f"Uploading batch {current_batch} of {total_batches}...")
        
        batch = chunks[i:i + batch_size]
        
        # Prepare data for upload
        data = []
        for chunk in batch:
            data.append({
                "content": chunk["content"],
                "embedding": chunk["embedding"],
                "page_number": chunk["page_number"],
                "section_title": chunk["section_title"],
                "aircraft_model": chunk["aircraft_model"],
                "document_id": chunk["document_id"],
                "system_category": chunk["system_category"],
                "metadata": {
                    "token_count": len(tokenizer.encode(chunk["content"]))
                }
            })
        
        # Try to upload with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Insert data into the Supabase table
                result = supabase.table("aircraft_manual").insert(data).execute()
                successful += len(result.data)
                print(f"Batch {current_batch}: Uploaded {len(result.data)} chunks successfully")
                break  # Exit retry loop on success
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Not the last attempt
                    print(f"Error uploading batch {current_batch}, attempt {attempt+1}: {str(e)}. Retrying...")
                    time.sleep(3)  # Pause before retry
                else:
                    # Last attempt failed
                    print(f"Failed to upload batch {current_batch} after {max_retries} attempts: {str(e)}")
                    failed += len(batch)
        
        # Brief pause between batches
        if i + batch_size < total_chunks:
            time.sleep(2)
    
    print(f"Upload complete! {successful} chunks uploaded successfully, {failed} failed")

def main():
    """Main process to extract, chunk, embed, and upload aircraft manual"""
    start_time = time.time()
    print(f"Processing aircraft manual: {PDF_PATH}")
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    print(f"Aircraft model: {AIRCRAFT_MODEL}")
    
    # Extract text from PDF
    pages_text = extract_text_from_pdf(PDF_PATH)
    
    if not pages_text:
        print("No content extracted from PDF. Process aborted.")
        return
    
    # Chunk the text
    chunks = chunk_text(pages_text)
    
    if not chunks:
        print("No chunks created. Process aborted.")
        return
    
    # Generate embeddings
    embedded_chunks = generate_embeddings(chunks)
    
    if not embedded_chunks:
        print("No embeddings generated. Process aborted.")
        return
    
    # Upload to Supabase
    upload_to_supabase(embedded_chunks)
    
    # Report processing time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Aircraft manual processing complete! Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()
