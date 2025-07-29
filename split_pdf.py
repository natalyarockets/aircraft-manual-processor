import fitz  # PyMuPDF
import re
from pathlib import Path
from collections import defaultdict

pdf_path = Path("/Users/n/Library/CloudStorage/GoogleDrive-natalya@bettersquawk.com/Shared drives/Engineering/Manuals and documentation/CJ3/CJ3 vector store approved and in use/AFM/CJ3-AFM.pdf")
doc = fitz.open(pdf_path)
output_folder = pdf_path.parent / "processed"
output_folder.mkdir(exist_ok=True)

# Match any ATA reference, e.g. 24-30-00 or 34-CONTENTS
chapter_regex = re.compile(r"\b(\d{2})[-–](?:\d{2}|[A-Z]+)(?:[-–]\d{2})?\b", re.IGNORECASE)

pages_by_chapter = defaultdict(list)

for i, page in enumerate(doc):
    height = page.rect.height
    footer_text = page.get_text("text")
    lines = footer_text.strip().splitlines()

    detected_chapter = None
    for line in lines[-5:]:  # Check the last 5 lines (footer zone)
        line = line.strip()
        if not line:
            continue
        match = chapter_regex.search(line)
        if match:
            detected_chapter = match.group(1)
            break

    if detected_chapter:
        pages_by_chapter[detected_chapter].append(i)

# Save one PDF per chapter
for chapter, page_nums in sorted(pages_by_chapter.items(), key=lambda x: int(x[0])):
    new_doc = fitz.open()
    for page_num in page_nums:
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    filename = f"CJ3 AFM chapter_{chapter}.pdf"
    output_path = output_folder / filename
    new_doc.save(output_path)
    print(f"Saved {filename} ({len(page_nums)} pages)")
