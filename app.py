import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Load API key
groq_api_key = os.getenv("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")

st.set_page_config(page_title="AI Support Bot")
st.title("💬 AI Customer Support Assistant")

# Upload files
uploaded_files = st.file_uploader(
    "Upload TXT or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

# Initialize models
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# Process files
if uploaded_files:
    all_docs = []

    for file in uploaded_files:
        path = f"temp_{file.name}"
        with open(path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        else:
            loader = PyPDFLoader(path)

        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Improved RAG
    def rag_chat(query):
        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])

        history = "\n".join(st.session_state.history[-5:])

        prompt = f"""
        You are a professional customer support assistant.

        Use ONLY the context below.
        If unsure, say "I don't know".

        Context:
        {context}

        Previous conversation:
        {history}

        Question: {query}

        Answer clearly and professionally.
        Also give confidence score (0 to 1).
        """

        response = llm.invoke(prompt).content

        # Parse confidence
        confidence = 0.8
        if "confidence" in response.lower():
            try:
                confidence = float(response.split()[-1])
            except:
                confidence = 0.5

        return response, confidence

    # Chat UI
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask your question")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        answer, confidence = rag_chat(user_input)

        # HITL fallback
        if confidence < 0.5:
            answer = "⚠️ I'm not confident. A human agent will assist you."

        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.session_state.history.append(user_input)
        st.session_state.history.append(answer)

        # Feedback
        col1, col2 = st.columns(2)
        with col1:
            st.button("👍 Helpful")
        with col2:
            st.button("👎 Not Helpful")

else:
    st.info("📄 Upload files to start chatting")
