import streamlit as st
import chromadb
import subprocess
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract

st.set_page_config(page_title="MRPL Sovereign AI Workbench")

# -------------------------------
# Load Embedding Model
# -------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Connect ChromaDB
# -------------------------------

client = chromadb.PersistentClient(path="../vector_db")
collection = client.get_or_create_collection("mrpl_documents")

# -------------------------------
# PDF Upload Section
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    save_path = f"../data/pdfs/{uploaded_file.name}"

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    reader = PdfReader(save_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # OCR fallback for scanned PDFs
    if len(text.strip()) == 0:

        st.info("📄 Scanned PDF detected. Running OCR...")

        images = convert_from_path(save_path)

        for image in images:
            text += pytesseract.image_to_string(image)

        st.success(f"OCR extracted {len(text)} characters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    st.write("Extracted characters:", len(text))
    st.write("Chunks created:", len(chunks))

    # Store chunks in ChromaDB
    for i, chunk in enumerate(chunks):

        embedding = model.encode(chunk).tolist()

        collection.add(
            ids=[f"{uploaded_file.name}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": uploaded_file.name}]
        )

    st.success(f"Indexed {len(chunks)} chunks into ChromaDB")

# -------------------------------
# Main App
# -------------------------------

st.title("🤖 MRPL Sovereign AI Workbench")

all_docs = collection.get()

pdf_files = []

if all_docs["metadatas"]:
    pdf_files = sorted(
        list(set([m["source"] for m in all_docs["metadatas"]]))
    )

if pdf_files:

    selected_pdf = st.selectbox(
        "Select Document",
        pdf_files
    )

    # Delete PDF
    if st.button("🗑 Delete Selected PDF"):

        data = collection.get()

        ids_to_delete = []

        for doc_id, metadata in zip(
            data["ids"],
            data["metadatas"]
        ):
            if metadata["source"] == selected_pdf:
                ids_to_delete.append(doc_id)

        if ids_to_delete:

            collection.delete(ids=ids_to_delete)

            st.success(f"Deleted {selected_pdf}")

            st.rerun()

    # -------------------------------
    # Chat History
    # -------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question")

    if question:

        query_embedding = model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"source": selected_pdf}
        )

        context = "\n\n".join(results["documents"][0])

        st.subheader("Retrieved Context")
        st.text(context[:3000])

        prompt = f"""
Answer ONLY from the provided information.

Information:
{context}

Question:
{question}

Answer:
"""

        st.info("🔍 Searching ChromaDB...")
        st.info("🧠 Generating answer with Qwen3...")

        with st.spinner("🤖 Thinking... Please wait..."):

            response = subprocess.run(
                ["ollama", "run", "qwen3:4b"],
                input=prompt,
                capture_output=True,
                text=True
            )

        answer = response.stdout

        if "done thinking." in answer:
            answer = answer.split("done thinking.")[-1].strip()

        # Save chat history

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Display current conversation

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(answer)

        st.subheader("Sources")

        for source in results["metadatas"][0]:
            st.write(source["source"])

else:

    st.warning("Upload at least one PDF to start asking questions.")
