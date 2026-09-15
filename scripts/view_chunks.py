import chromadb

client = chromadb.PersistentClient(path="../vector_db")
collection = client.get_collection("resume_chunks")

data = collection.get()

for i, doc in enumerate(data["documents"]):
    print(f"\n===== CHUNK {i} =====\n")
    print(doc)
