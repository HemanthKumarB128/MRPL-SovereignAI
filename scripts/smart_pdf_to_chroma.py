from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

pdf_folder = "../data/pdfs"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="../vector_db")

collection = client.get_or_create_collection("mrpl_documents")

chunk_id = 0

for pdf_file in os.listdir(pdf_folder):

    if pdf_file.endswith(".pdf"):

        print(f"\nProcessing: {pdf_file}")

        reader = PdfReader(os.path.join(pdf_folder, pdf_file))

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        chunks = splitter.split_text(text)

        for chunk in chunks:

            embedding = model.encode(chunk).tolist()

            collection.add(
                ids=[f"chunk_{chunk_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": pdf_file}]
            )

            chunk_id += 1

        print(f"Stored {len(chunks)} chunks from {pdf_file}")

print("\nAll PDFs processed successfully!")
