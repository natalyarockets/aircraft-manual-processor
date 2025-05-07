import os
import json
import time
import numpy as np
from tqdm import tqdm
from supabase import create_client
from config import *

# OpenAI
from openai import OpenAI
import tiktoken

# Hugging Face
from transformers import AutoTokenizer, AutoModel
import torch

# Supabase init
supabase = create_client(SUPABASE_VS_URL, SUPABASE_VS_KEY)

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
    print("Uploading to Supabase...")
    batch_size = 50
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
                "embedding_model": chunk["embedding_model"],
                "metadata": {
                    "token_count": chunk.get("token_count", 0)
                }
            })
        try:
            res = supabase.table("aircraft_manual").insert(data).execute()
            print(f"Uploaded batch {i // batch_size + 1}")
        except Exception as e:
            print(f"Upload failed for batch {i // batch_size + 1}: {e}")


def main():
    print(f"Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_PROVIDER})")
    print(f"Loading chunks from: {RAW_CHUNKS_PATH}")
    chunks = load_chunks(RAW_CHUNKS_PATH)

    if EMBEDDING_PROVIDER == "openai":
        embedded = embed_with_openai(chunks, EMBEDDING_MODEL)
    elif EMBEDDING_PROVIDER == "huggingface":
        embedded = embed_with_huggingface(chunks, EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")

    if not embedded:
        print("No embeddings generated. Exiting.")
        return

    save_embedded_chunks(embedded, EMBEDDED_CHUNKS_PATH)
    upload_to_supabase(embedded)


if __name__ == "__main__":
    main()
