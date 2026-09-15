import chromadb
import subprocess
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

question = input("Ask a question: ")

query_embedding = model.encode(question).tolist()

client = chromadb.PersistentClient(path="../vector_db")

print(client.list_collections())

collection = client.get_collection("mrpl_documents")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\n=== SOURCES ===")
print(results["metadatas"][0])

context = "\n".join(results["documents"][0])
print("\n=== RETRIEVED CONTEXT ===\n")
print(context)
print("\n=========================\n")
prompt = f"""
Answer the question using only the provided information.

Information:
{context}

Question:
{question}

Answer:
"""

print("\nThinking...\n")

response = subprocess.run(
    ["ollama", "run", "qwen3:4b"],
    input=prompt,
    text=True,
    capture_output=True
)

print(response.stdout)
