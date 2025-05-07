import os
import json
import time
import numpy as np
from tqdm import tqdm
from supabase import create_client, Client
from config import *

# OpenAI
from openai import OpenAI
import tiktoken

# Hugging Face
from transformers import AutoTokenizer, AutoModel
import torch

# Supabase init
#print(SUPABASE_VS_URL)
supabase: Client = create_client(SUPABASE_VS_URL, SUPABASE_VS_KEY)

# OpenAI init
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Tokenizer for OpenAI
openai_tokenizer = tiktoken.get_encoding("cl100k_base")


def load_chunks(path):
    with open(path, "r") as f:
        return json.load(f)


def save_embedded_chunks(chunks, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved embedded chunks to {path}")


def embed_with_openai(chunks, model):
    print(f"Embedding with OpenAI: {model}")
    embedded = []
    for chunk in tqdm(chunks):
        try:
            response = openai_client.embeddings.create(
                model=model,
                input=chunk["content"]
            )
            embedding = response.data[0].embedding
            chunk["embedding"] = embedding
            chunk["embedding_model"] = model
            embedded.append(chunk)
        except Exception as e:
            print(f"[OpenAI Error] Page {chunk['page_number']}: {e}")
    return embedded


def embed_with_huggingface(chunks, model_name):
    print(f"Embedding with Hugging Face: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)

    embedded = []
    for chunk in tqdm(chunks):
        try:
            inputs = tokenizer(chunk["content"], return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                model_output = model(**inputs)
                embeddings = model_output.last_hidden_state[:, 0, :]
                vector = embeddings.squeeze().cpu().numpy().tolist()

            chunk["embedding"] = vector
            chunk["embedding_model"] = model_name
            embedded.append(chunk)
        except Exception as e:
            print(f"[HF Error] Page {chunk['page_number']}: {e}")
    return embedded


def upload_to_supabase(chunks):
    print(f"Uploading {len(chunks)} chunks to Supabase table: {TABLE_NAME}")
    batch_size = 50
    successful = 0
    failed = 0

    # First, try to retrieve the table structure to verify connection
    try:
        # This will only work if you have the right permissions - for debugging
        print("Testing Supabase connection...")
        response = supabase.table(TABLE_NAME).select("*").limit(1).execute()
        print(f"Connection successful. Table exists.")
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
        print(f"URL: {SUPABASE_VS_URL}")
        # Don't print the actual key for security
        print(f"Key valid: {bool(SUPABASE_VS_KEY and SUPABASE_VS_KEY.startswith('eyJ'))}")
        
    # Print dimensions of first embedding to verify
    if chunks and "embedding" in chunks[0]:
        print(f"Embedding dimensions: {len(chunks[0]['embedding'])}")
        
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
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
                "document_type": DOCUMENT_TYPE,
                "metadata": {
                    "token_count": chunk.get("token_count", 0)
                }
            })

        try:
            print(f"Attempting to insert batch {i // batch_size + 1}...")
            response = supabase.table(TABLE_NAME).insert(data).execute()
            
            if hasattr(response, "data") and response.data:
                print(f"✅ Batch {i // batch_size + 1}: Inserted {len(response.data)} records")
                successful += len(response.data)
            else:
                print(f"❌ Batch {i // batch_size + 1}: Insert returned no data")
                # Try to get more details about the response
                print(f"Response object: {response}")
                if hasattr(response, "error"):
                    print(f"Error: {response.error}")
                failed += len(batch)
                
        except Exception as e:
            print(f"🔥 Batch {i // batch_size + 1} failed with exception: {str(e)}")
            # Print the first data item to see structure (without the full embedding)
            if data:
                debug_item = data[0].copy()
                if "embedding" in debug_item:
                    debug_item["embedding"] = f"[{len(debug_item['embedding'])} dimensions]"
                print(f"Sample data structure: {debug_item}")
            failed += len(batch)

    print(f"\nUpload complete: {successful} chunks uploaded successfully, {failed} failed.")



def main():
    print(f"Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_PROVIDER})")
    print(f"Supabase table: {TABLE_NAME}")

    if os.path.exists(EMBEDDED_CHUNKS_PATH):
        print(f"🔁 Found embedded chunks at {EMBEDDED_CHUNKS_PATH}. Skipping embedding.")
        embedded = load_chunks(EMBEDDED_CHUNKS_PATH)
    else:
        print("🧠 No embedded chunks found. Running embedding now...")
        chunks = load_chunks(RAW_CHUNKS_PATH)

        if EMBEDDING_PROVIDER == "openai":
            embedded = embed_with_openai(chunks, EMBEDDING_MODEL)
        elif EMBEDDING_PROVIDER == "huggingface":
            embedded = embed_with_huggingface(chunks, EMBEDDING_MODEL)
        else:
            raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")

        save_embedded_chunks(embedded, EMBEDDED_CHUNKS_PATH)

    upload_to_supabase(embedded)



if __name__ == "__main__":
    main()
