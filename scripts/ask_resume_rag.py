import chromadb
import subprocess
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

context = results["documents"][0][0]

prompt = f"""
Answer the question using the provided resume information.

Resume Information:
{context}

Question:
{question}

Answer:
"""

result = subprocess.run(
    ["ollama", "run", "qwen3:4b", prompt],
    capture_output=True,
    text=True
)

print(result.stdout)
