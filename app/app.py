import streamlit as st
import chromadb
import subprocess
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="MRPL Sovereign AI Workbench")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:
    save_path = f"../data/pdfs/{uploaded_file.name}"

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")


st.title("🤖 MRPL Sovereign AI Workbench")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="../vector_db")
collection = client.get_collection("mrpl_documents")

question = st.text_input("Ask a question")

if question:

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
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

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for source in results["metadatas"][0]:
        st.write(source["source"])
