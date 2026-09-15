from pypdf import PdfReader

pdf_path = "../data/pdfs/sample.pdf"

reader = PdfReader(pdf_path)

for page in reader.pages:
    print(page.extract_text())

