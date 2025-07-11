import os
from dotenv import load_dotenv

load_dotenv()  # This reads from your .env file

# Aircraft and PDF config
AIRCRAFT_MODEL = "SR22G5"
PDF_PATH = f"/Users/n/Library/CloudStorage/GoogleDrive-natalya@bettersquawk.com/Shared drives/Engineering/Manuals and documentation/DA42/DA42 vector store approved and in use/DA42-NG-AMM.pdf"
DOCUMENT_TYPE = "maintenance_manual"  # You can change this if you want

# Embedding config
EMBEDDING_PROVIDER = "openai" # or "huggingface" 
EMBEDDING_MODEL = "text-embedding-3-small" # or "BAAI/bge-base-en-v1.5"
TABLE_NAME = f"aircraft_manual_{EMBEDDING_MODEL.replace('/', '_').replace('-', '_').replace('.', '_')}".lower()


# Output paths
RAW_CHUNKS_PATH = f"data/{AIRCRAFT_MODEL}/chunks_raw.json"
EMBEDDED_CHUNKS_PATH = f"data/{AIRCRAFT_MODEL}/chunks_embedded_{EMBEDDING_MODEL.replace('/', '_')}.json"

# API Keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_VS_URL = os.getenv("SUPABASE_VS_URL")
SUPABASE_VS_KEY = os.getenv("SUPABASE_VS_KEY")