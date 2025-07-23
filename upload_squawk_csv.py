##upload_squawk_csv.py

import csv
import json
from datetime import datetime
import openai
from supabase import create_client
from config import OPENAI_API_KEY, SUPABASE_VS_URL, SUPABASE_VS_KEY

# Config
openai.api_key = OPENAI_API_KEY
supabase = create_client(SUPABASE_VS_URL, SUPABASE_VS_KEY)
TABLE_NAME = "squawk_text_embedding_3_small"
CSV_FILE = "/Users/n/Library/CloudStorage/GoogleDrive-natalya@bettersquawk.com/Shared drives/Engineering/Customer data/Pacific Coast Jets/Analytics Aircraft Discrepancies gm1.csv"  # path to your CSV file

def embed_text(text: str) -> list[float]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

def parse_timestamp(text: str) -> datetime | None:
    for fmt in ("%m/%d/%y %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%y %H:%M:%S"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except:
            continue
    return None

def iso_or_none(text: str) -> str | None:
    if not text or text.strip().lower() in {"n/a", "na"}:
        return None
    ts = parse_timestamp(text)
    return ts.isoformat() if ts else None

def load_and_upload():
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            issue = row.get("Issue", "").strip()
            corrective = row.get("Corrective action", "").strip()
            issue_embedding = embed_text(issue)
            corrective_embedding = embed_text(corrective)

            record = {
                "discrepancy_id": row.get("Discrepancy ID"),
                "org": "pcj",  # Replace or pass dynamically
                "tail_number": row.get("Aircraft"),
                "aircraft_type": row.get("Aircraft Serial"),
                "aircraft": row.get("Aircraft"),
                "aircraft_serial": row.get("Aircraft Serial"),
                "logged_on": iso_or_none(row.get("Logged on", "")),
                "issue": issue,
                "corrective_action": corrective,
                "status": row.get("Status"),
                "mel_category": row.get("MEL category"),
                "mel_location": row.get("MEL location"),
                "mel_company": row.get("MEL company"),
                "signoff_by": row.get("Sign-off by"),
                "signoff_on": iso_or_none(row.get("Sign-off on", "")),
                "signoff_license": row.get("Sign-off license"),
                "work_completed_on": iso_or_none(row.get("Work completed on", "")),
                "issue_embedding_vector": issue_embedding,
                "corrective_embedding_vector": corrective_embedding,
            }

            print(f"Inserting: {record['discrepancy_id']}")
            # preview = record.copy()
            # preview["issue_embedding_vector"] = preview["issue_embedding_vector"][:5]
            # preview["corrective_embedding_vector"] = preview["corrective_embedding_vector"][:5]
            # print(json.dumps(preview, indent=2))
            # break

            supabase.table(TABLE_NAME).insert(record).execute()

if __name__ == "__main__":
    load_and_upload()