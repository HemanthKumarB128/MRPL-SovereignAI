import chromadb

client = chromadb.PersistentClient(path="../vector_db")

print("Collections found:")

for collection in client.list_collections():
    print("\nCollection:", collection.name)

    col = client.get_collection(collection.name)

    data = col.get()

    print("Documents:", len(data["documents"]))

    for i, doc in enumerate(data["documents"][:3]):
        print(f"\n--- Document {i+1} ---")
        print(doc[:300])
