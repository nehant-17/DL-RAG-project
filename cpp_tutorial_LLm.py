import streamlit as st 
import os 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from lanfchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama 

# __________________________
# Page Config 
#____________________________

st.title("C++ RAG Chatbot with Ollama (gemma2:9b)")
#st.set_page_config(page_title="Langchain RAG Demo with gemma2:9b", page_icon=":material/smart_toy")

# ___________________________
# Load & Process Data
# ___________________________

def load_vectorstore():
    loader = TextLoader("cpp.txt",encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 20
    )
    final_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name = "all-MiniLLM-L6-v2"
    )

    db = FAISS.from_documents(final_docs, embeddings)
    return db


# _____________________________
# Load LLM (Ollama)
# _____________________________

llm = Ollama(model = "gemma2:9b")


# ------------------------
#Chat Interface 
# --------------------------
user_question = st.text_input("Ask a question about C++ programming : ")

if user_question:
    with st.spinner("Ask a question about C++ : "):
        docs = db.similarity_search(user_question)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using only the context below.
        Context: 
        {context}
        Question:
        {user_question}

        Answer:
        """

# You are a helpful C++ programming assistant.
# Answer the user's questiom using the retrieved context."""
        response = llm.invoke(prompt)
        st.subheader("Answer : ")
        st.write(response)