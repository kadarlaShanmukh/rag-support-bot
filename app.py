import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from langgraph.graph import StateGraph, END
from typing import TypedDict

# -------------------------------
# Load API Key
# -------------------------------
groq_api_key = os.getenv("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")

# -------------------------------
#  UI Setup
# -------------------------------
st.set_page_config(page_title="AI Support Bot")
st.title("💬 AI Customer Support Assistant")

# -------------------------------
# File Upload
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# -------------------------------
# Initialize Models
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_api_key
)

# -------------------------------
# Graph State Definition
# -------------------------------
class State(TypedDict):
    question: str
    context: str
    answer: str
    confidence: float


# -------------------------------
# Node 1: Retrieve
# -------------------------------
def retrieve(state):
    docs = retriever.invoke(state["question"])
    context = "\n".join([doc.page_content for doc in docs])
    return {"context": context}


# -------------------------------
# Node 2: Generate Answer
# -------------------------------
def generate(state):
    prompt = f"""
    You are a professional customer support assistant.

    Use ONLY the context below to answer.
    If you don't know, say "I don't know".

    Context:
    {state['context']}

    Question: {state['question']}

    Also provide confidence between 0 and 1.
    """

    response = llm.invoke(prompt).content

    # Simple confidence logic (basic version)
    confidence = 0.8
    if "don't know" in response.lower():
        confidence = 0.3

    return {
        "answer": response,
        "confidence": confidence
    }


# -------------------------------
#  Node 3: Decision (HITL)
# -------------------------------
def decision(state):
    if state["confidence"] < 0.6:
        return "human"
    return "end"


# -------------------------------
#  Node 4: Human Escalation
# -------------------------------
def human_node(state):
    return {
        "answer": "⚠️ I'm not confident about this. Escalating to human support."
    }


# -------------------------------
# process Uploaded Files
# -------------------------------
if uploaded_files:
    all_docs = []

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

    # -------------------------------
    #  Chunking
    # -------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    # -------------------------------
    #  Vector DB
    # -------------------------------
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # -------------------------------
    #  Build LangGraph
    # -------------------------------
    graph = StateGraph(State)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("human", human_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "generate")

    graph.add_conditional_edges(
        "generate",
        decision,
        {
            "human": "human",
            "end": END
        }
    )

    app_graph = graph.compile()

    # -------------------------------
    # Chat UI
    # -------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask your question")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        result = app_graph.invoke({
            "question": user_input
        })

        answer = result["answer"]

        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("📄 Please upload documents to start")
