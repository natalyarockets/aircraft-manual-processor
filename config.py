#config.py

import os
from dotenv import load_dotenv

load_dotenv()  # This reads from your .env file

## For processing and embedding manuals
# !Set these each time!
# Aircraft and PDF config
AIRCRAFT_MODEL = "DA42"
PDF_PATH = f"/Users/n/Library/CloudStorage/GoogleDrive-natalya@bettersquawk.com/Shared drives/Engineering/Manuals and documentation/DA42/DA42 vector store approved and in use/DA42-NG-AFM.pdf"
DOCUMENT_TYPE = "AFM"  # "maintenance manual" or "POH" or "training manual" or "AFM" or "operating manual" or "specs"

# Embedding config
EMBEDDING_PROVIDER = "openai" # "huggingface" or "openai"

if EMBEDDING_PROVIDER == "huggingface":
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Hugging Face model
else:
    EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model

TABLE_NAME = f"aircraft_manual_{EMBEDDING_MODEL.replace('/', '_').replace('-', '_').replace('.', '_')}".lower()

# Output paths
# config.py
RAW_CHUNKS_PATH = f"data/{AIRCRAFT_MODEL}/{DOCUMENT_TYPE}_chunks_raw.json"
EMBEDDED_CHUNKS_PATH = f"data/{AIRCRAFT_MODEL}/{DOCUMENT_TYPE}_chunks_embedded_{EMBEDDING_MODEL.replace('/', '_')}.json"

# API Keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_VS_URL = os.getenv("SUPABASE_VS_URL")
SUPABASE_VS_KEY = os.getenv("SUPABASE_VS_KEY")