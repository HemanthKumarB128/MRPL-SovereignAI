from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "This is a test document for MRPL Sovereign AI."

embedding = model.encode(text)

print("Embedding length:", len(embedding))
print("First 10 values:")
print(embedding[:10])
