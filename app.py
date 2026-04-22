import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# API key from Streamlit secrets
groq_api_key = os.getenv("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")

st.set_page_config(page_title="AI Support Bot")
st.title("💬 AI Customer Support Assistant")

# Upload multiple files
uploaded_files = st.file_uploader(
    "Upload TXT or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

# Initialize
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

if uploaded_files:
    all_docs = []

    # Save and load files
    for file in uploaded_files:
        file_path = f"temp_{file.name}"

        with open(file_path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            loader = PyPDFLoader(file_path)

        docs = loader.load()
        all_docs.extend(docs)

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_docs)

    # Vector DB
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # RAG function
    def rag_chat(query):
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer only from this context:

        {context}

        Question: {query}
        """

        response = llm.invoke(prompt)
        return response.content

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask your question")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        answer = rag_chat(user_input)

        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("📄 Upload one or more files to start chatting")
