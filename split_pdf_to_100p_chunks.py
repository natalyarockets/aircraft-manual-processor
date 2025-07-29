import fitz  # PyMuPDF
from pathlib import Path

# === Config ===
pdf_path = Path("/Users/n/Library/CloudStorage/GoogleDrive-natalya@bettersquawk.com/Shared drives/Engineering/Manuals and documentation/CJ3/CJ3 vector store approved and in use/AFM/CJ3-AFM.pdf")
output_folder = pdf_path.parent / "processed"
output_folder.mkdir(exist_ok=True)

doc = fitz.open(pdf_path)
total_pages = len(doc)
chunk_size = 100

# === Split into chunks ===
for start_page in range(0, total_pages, chunk_size):
    end_page = min(start_page + chunk_size - 1, total_pages - 1)
    chunk_doc = fitz.open()
    chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)

    chunk_filename = f"{pdf_path.stem}_pages_{start_page+1}-{end_page+1}.pdf"
    output_path = output_folder / chunk_filename
    chunk_doc.save(output_path)
    print(f"Saved {chunk_filename} ({end_page - start_page + 1} pages)")
