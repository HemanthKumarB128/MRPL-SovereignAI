import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

question = input("Ask a question: ")

query_embedding = model.encode(question).tolist()

client = chromadb.PersistentClient(path="../vector_db")

collection = client.get_collection("resume_chunks")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nMost Relevant Chunks:\n")

for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- Chunk {i+1} ---\n")
    print(doc)
