import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# API Key
os.environ["GROQ_API_KEY"] = "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"

st.set_page_config(page_title="AI Support Bot")
st.title("💬 AI Customer Support Assistant")

# Upload file
uploaded_file = st.file_uploader("Upload a file")

# Initialize embeddings + model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# Process uploaded file
if uploaded_file:
    with open("temp.txt", "wb") as f:
        f.write(uploaded_file.read())

    loader = TextLoader("temp.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

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
    st.info("📄 Please upload a file to start chatting")