import fitz  # PyMuPDF
import re
from pathlib import Path
from collections import defaultdict

pdf_path = Path("/Users/n/Library/CloudStorage/GoogleDrive-natalya@bettersquawk.com/Shared drives/Engineering/Manuals and documentation/DA40/60201-r10-DA40-AMM-complete.pdf")
doc = fitz.open(pdf_path)
output_folder = pdf_path.parent / "processed"
output_folder.mkdir(exist_ok=True)

# Match any ATA reference, e.g. 24-30-00 or 34-CONTENTS
chapter_regex = re.compile(r"\b(\d{2})[-–](?:\d{2}|[A-Z]+)(?:[-–]\d{2})?\b", re.IGNORECASE)
footer_threshold = 0.85  # bottom 15% of the page

pages_by_chapter = defaultdict(list)

for i, page in enumerate(doc):
    blocks = page.get_text("dict")["blocks"]
    height = page.rect.height

    detected_chapter = None
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                y = span["bbox"][1]
                text = span["text"].strip()
                if y > height * footer_threshold:
                    match = chapter_regex.search(text)
                    if match:
                        detected_chapter = match.group(1)
    if detected_chapter:
        pages_by_chapter[detected_chapter].append(i)

# Save one PDF per chapter
for chapter, page_nums in sorted(pages_by_chapter.items(), key=lambda x: int(x[0])):
    new_doc = fitz.open()
    for page_num in page_nums:
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    filename = f"DA 40 AMM chapter_{chapter}.pdf"
    output_path = output_folder / filename
    new_doc.save(output_path)
    print(f"Saved {filename} ({len(page_nums)} pages)")
