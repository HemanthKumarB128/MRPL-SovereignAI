import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

question = input("Ask a question: ")

query_embedding = model.encode(question).tolist()

client = chromadb.PersistentClient(path="../vector_db")
collection = client.get_collection("resume_docs")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print("\nRelevant Document Content:\n")
print(results["documents"][0][0][:2000])
