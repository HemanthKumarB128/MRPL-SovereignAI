from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader("../data/pdfs/sample.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print(f"Total Smart Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n===== CHUNK {i+1} =====\n")
    print(chunk)
