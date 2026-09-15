import chromadb
import subprocess
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

question = input("Ask a question: ")

query_embedding = model.encode(question).tolist()

client = chromadb.PersistentClient(path="../vector_db")
collection = client.get_collection("resume_smart_chunks")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

context = "\n\n".join(results["documents"][0])

prompt = f"""
Answer the question using only the provided information.

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

print("\nAnswer:\n")
print(response.stdout)
