import re
import jsonlines
import json
from pathlib import Path
from pypdf import PdfReader
from rich.progress import track
from rich import print

# Read the PDF file
pdf_path = "./data/UG Policy.pdf"
reader = PdfReader(pdf_path)
pages = reader.pages[1:]

# Extract text from all pages
contents = "\n".join([page.extract_text() for page in pages])

# List to hold bold texts
bold_texts = []

# Visitor function to capture bold text
def visit_body(text, cm, tm, font_dict, font_size):
    if (
        len(text.strip()) > 0
        and font_dict is not None
        and font_dict["/BaseFont"] == "/BAAAAA+ArialMT"
    ):
        bold_texts.append(text.strip())

# Extract bold text from each page
for page in track(pages, description="Extracting text from PDF"):
    page.extract_text(visitor_text=visit_body)

# Debug: Print number of bold texts extracted
print(f"Number of bold texts extracted: {len(bold_texts)}")

# Using bold texts as markers to segment the contents
start_indexes = [contents.find(text) for text in bold_texts]

# Add the end of the document as the last index
start_indexes.append(len(contents))

# Extract data based on start indexes
extracted_data = []
for i in track(range(len(start_indexes) - 1), description="Retrieving useful content from extracted text"):
    start = start_indexes[i]
    end = start_indexes[i + 1]
    extracted_data.append({"doc": contents[start:end]})

# Ensure the output directory exists
output_dir = Path("./extracted_data")
output_dir.mkdir(parents=True, exist_ok=True)

# Write extracted data to a JSONL file
output_path = output_dir / "policy.jsonl"
print(f"Writing extracted text to [bold]'{output_path}'")
with jsonlines.open(output_path, mode="w") as writer:
    writer.write_all(extracted_data)

# Debug: Confirm completion
print("Data extraction and writing completed successfully.")
