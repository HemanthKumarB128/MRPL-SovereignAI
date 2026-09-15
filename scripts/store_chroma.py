import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "This is a test document for MRPL Sovereign AI."

embedding = model.encode(text).tolist()

client = chromadb.PersistentClient(path="../vector_db")

collection = client.get_or_create_collection("mrpl_docs")

collection.add(
    ids=["doc1"],
    embeddings=[embedding],
    documents=[text]
)

print("Document stored successfully!")

