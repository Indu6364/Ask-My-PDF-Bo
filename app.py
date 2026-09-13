import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

# Gemini API Key
genai.configure(
API_KEY = "your_actual_api_key"
)

st.set_page_config(
    page_title="LegalMind AI",
    layout="wide"
)

st.title("LegalMind AI (Multi-PDF RAG Chatbot)")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    # Read PDFs
    text = ""

    for pdf in uploaded_files:

        reader = PdfReader(pdf)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    # Embedding Model
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = embedding_model.encode(chunks)

    # FAISS Index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings)
    )

    st.success(
        f"{len(uploaded_files)} PDFs processed. {len(chunks)} chunks stored."
    )

    # Display previous messages
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(
                msg["content"]
            )

    # Question Input
    question = st.chat_input(
        "Ask a question about the PDFs"
    )

    if question:

        # Show User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # Convert Question to Embedding
        question_embedding = embedding_model.encode(
            [question]
        )

        # Retrieve Top 10 Chunks
        k = min(10, len(chunks))

        distances, indices = index.search(
            np.array(question_embedding),
            k
        )

        relevant_chunks = []

        for idx in indices[0]:

            chunk = chunks[idx]

            if chunk not in relevant_chunks:

                relevant_chunks.append(
                    chunk
                )

        context = "\n\n".join(
            relevant_chunks[:10]
        )

        try:

            model = genai.GenerativeModel(
                "models/gemini-3.6-flash"
            )

            prompt = f"""
You are a legal document assistant.

Answer ONLY using the context below.

If information appears in any retrieved chunk,
use it.

Context:
{context}

Question:
{question}

Provide concise answers.

If information is missing, write:
Answer not found in document.
"""

            response = model.generate_content(
                prompt
            )

            answer = response.text

        except Exception as e:

            answer = f"""
Gemini Error:
{e}

Showing retrieved context instead.
"""

        # Store Assistant Response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Show Assistant Response
        with st.chat_message("assistant"):

            st.markdown(answer)

        # Retrieved Chunks
        st.subheader(
            "Retrieved Chunks"
        )

        for i, idx in enumerate(indices[0]):

            st.write(
                f"Chunk {i+1}"
            )

            st.write(
                chunks[idx]
            )

            st.write(
                f"Distance Score: {distances[0][i]:.4f}"
            )

            st.divider()