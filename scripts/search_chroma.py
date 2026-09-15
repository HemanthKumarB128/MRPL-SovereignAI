import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "AI system"

query_embedding = model.encode(query).tolist()

client = chromadb.PersistentClient(path="../vector_db")

collection = client.get_collection("mrpl_docs")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print(results["documents"][0][0])
