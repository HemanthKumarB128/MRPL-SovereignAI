from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# Read PDF
reader = PdfReader("../data/pdfs/sample.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# Split text into chunks
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

print(f"Total Chunks Created: {len(chunks)}")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="../vector_db")

collection = client.get_or_create_collection("resume_chunks")

# Store chunks
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()

    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk]
    )

print("All chunks stored successfully!")
