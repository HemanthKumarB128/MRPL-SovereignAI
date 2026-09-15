from pypdf import PdfReader
import os

pdf_folder = "../data/pdfs"

for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        print(f"\nProcessing: {pdf_file}")

        reader = PdfReader(os.path.join(pdf_folder, pdf_file))

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        print(f"Extracted {len(text)} characters")
