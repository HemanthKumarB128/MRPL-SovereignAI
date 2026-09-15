from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

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

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="../vector_db")

collection = client.get_or_create_collection("resume_smart_chunks")

for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()

    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk]
    )

print(f"{len(chunks)} smart chunks stored successfully!")
