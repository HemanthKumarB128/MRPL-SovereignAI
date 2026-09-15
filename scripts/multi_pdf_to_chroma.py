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

doc_id = 0

for pdf_file in os.listdir(pdf_folder):

    if pdf_file.endswith(".pdf"):

        print(f"Processing {pdf_file}")

        reader = PdfReader(os.path.join(pdf_folder, pdf_file))

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        chunks = splitter.split_text(text)

        for chunk in chunks:

            embedding = model.encode(chunk).tolist()

            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": pdf_file}]
            )

            doc_id += 1

print("All PDFs stored successfully!")
