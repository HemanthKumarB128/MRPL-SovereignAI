from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

reader = PdfReader("../data/pdfs/sample.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(text).tolist()

client = chromadb.PersistentClient(path="../vector_db")
collection = client.get_or_create_collection("resume_docs")

collection.add(
    ids=["resume1"],
    embeddings=[embedding],
    documents=[text]
)

print("Resume stored in ChromaDB successfully!")
