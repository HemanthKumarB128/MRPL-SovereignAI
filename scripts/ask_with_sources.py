import chromadb
import subprocess
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="../vector_db")
collection = client.get_collection("mrpl_documents")

question = input("Ask a question: ")

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

context = "\n\n".join(results["documents"][0])

print("\n===== SOURCES =====\n")

for i, doc in enumerate(results["documents"][0]):
    print(f"\nSource {i+1}:")
    print(doc[:300])

prompt = f"""
Answer ONLY from the provided information.

Information:
{context}

Question:
{question}

Answer:
"""

response = subprocess.run(
    ["ollama", "run", "qwen3:4b"],
    input=prompt,
    text=True,
    capture_output=True
)

print("\n===== ANSWER =====\n")
print(response.stdout)
