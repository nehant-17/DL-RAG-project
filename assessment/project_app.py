import io
import streamlit as st

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM


# --------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM PAGE STYLE
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --app-bg: #0b1020;
        --panel-bg: #121a2b;
        --panel-border: #28344b;
        --text-main: #edf2ff;
        --text-muted: #a8b4cc;
        --accent: #7c9cff;
        --accent-hover: #9bb3ff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at top left, rgba(65, 91, 160, 0.20), transparent 38%),
            linear-gradient(145deg, #0b1020 0%, #10182a 55%, #0b1020 100%);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.035em;
    }

    h1 {
        font-weight: 800 !important;
        color: #f5f7ff !important;
    }

    h2, h3 {
        color: #e3eaff !important;
        font-weight: 700 !important;
    }

    p, label, .stMarkdown, .stCaption {
        color: var(--text-muted);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10182a 0%, #0d1424 100%);
        border-right: 1px solid var(--panel-border);
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stNumberInput"] input {
        background-color: #121a2b;
        color: var(--text-main);
        border-color: #34425d;
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed #52658c;
    }

    .stButton > button {
        border: 1px solid #7c9cff;
        border-radius: 10px;
        background: linear-gradient(135deg, #7896ff, #6478e8);
        color: #081020;
        font-weight: 700;
        padding: 0.55rem 1rem;
        transition: all 160ms ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #a2b7ff, #8194ff);
        border-color: #a2b7ff;
        color: #081020;
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: rgba(18, 26, 43, 0.78);
        border: 1px solid var(--panel-border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--panel-border);
        border-radius: 12px;
        background: rgba(18, 26, 43, 0.45);
    }

    hr {
        border-color: var(--panel-border);
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📚 Research Paper Assistant")
st.caption(
    "Explore your research papers with retrieval-augmented generation. "
    "Upload PDFs or text files, then ask questions grounded in their content."
)


# --------------------------------------------------
# 2. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    model_name = st.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2"]
    )

    ollama_model = st.selectbox(
        "Language Model",
        ["gemma2:9b"]
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=1500,
        value=800,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=300,
        value=100,
        step=50
    )

    top_k = st.slider(
        "Relevant Chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    response_mode = st.radio(
        "Answering Mode",
        [
            "Hybrid (Papers + Model Knowledge)",
            "Research Papers Only",
            "General Knowledge Only"
        ],
        index=0,
        help=(
            "Hybrid uses relevant paper passages and the model's pretrained knowledge. "
            "Research Papers Only restricts answers to retrieved passages. "
            "General Knowledge Only answers without searching uploaded papers."
        )
    )

    st.divider()

    st.info(
        "Embeddings retrieve relevant text. "
        "Gemma generates the answer using that text."
    )


# --------------------------------------------------
# 3. INITIALIZE EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embeddings(name):
    return HuggingFaceEmbeddings(
        model_name=name
    )


# --------------------------------------------------
# 4. INITIALIZE OLLAMA MODEL
# --------------------------------------------------

@st.cache_resource
def load_llm(name):
    return OllamaLLM(
        model=name,
        temperature=0
    )


# --------------------------------------------------
# 5. LOAD DOCUMENTS
# --------------------------------------------------

def load_uploaded_documents(uploaded_files):

    documents = []

    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()

        try:

            # PDF FILE
            if file_name.lower().endswith(".pdf"):

                pdf_reader = PdfReader(
                    io.BytesIO(file_bytes)
                )

                for page_number, page in enumerate(
                    pdf_reader.pages,
                    start=1
                ):

                    page_text = page.extract_text() or ""

                    if page_text.strip():

                        documents.append(
                            Document(
                                page_content=page_text,
                                metadata={
                                    "source": file_name,
                                    "page": page_number
                                }
                            )
                        )

            # TXT FILE
            elif file_name.lower().endswith(".txt"):

                text = file_bytes.decode("utf-8-sig")

                if text.strip():

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_name
                            }
                        )
                    )

        except Exception as error:

            st.error(
                f"Could not read {file_name}: {error}"
            )

    return documents


# --------------------------------------------------
# 6. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

def split_documents(documents, size, overlap):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap
    )

    return splitter.split_documents(documents)


# --------------------------------------------------
# 7. CREATE FAISS VECTOR DATABASE
# --------------------------------------------------

def create_vector_database(chunks, embeddings):

    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )


# --------------------------------------------------
# 8. UPLOAD DOCUMENTS
# --------------------------------------------------

st.subheader("📂 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


# --------------------------------------------------
# 9. PROCESS DOCUMENTS
# --------------------------------------------------

if uploaded_files:

    if st.button(
        "Process Documents",
        type="primary"
    ):

        with st.spinner(
            "Reading documents and creating embeddings..."
        ):

            try:

                raw_documents = load_uploaded_documents(
                    uploaded_files
                )

                if not raw_documents:
                    st.error(
                        "No readable text was found in the uploaded files."
                    )
                    st.stop()

                chunks = split_documents(
                    raw_documents,
                    chunk_size,
                    chunk_overlap
                )

                embeddings = load_embeddings(
                    model_name
                )

                vector_db = create_vector_database(
                    chunks,
                    embeddings
                )

                st.session_state["vector_db"] = vector_db
                st.session_state["chunks"] = chunks

                st.session_state["processed_files"] = [
                    file.name for file in uploaded_files
                ]

                st.session_state["embedding_model"] = model_name
                st.session_state["chunk_size"] = chunk_size
                st.session_state["chunk_overlap"] = chunk_overlap

                st.success(
                    "Documents processed successfully!"
                )

            except Exception as error:

                st.error(
                    f"Document processing failed: {error}"
                )


# --------------------------------------------------
# 10. DISPLAY DATABASE INFORMATION
# --------------------------------------------------

if "vector_db" in st.session_state:

    st.divider()

    st.subheader("📊 Vector Database Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Documents",
        len(st.session_state["processed_files"])
    )

    col2.metric(
        "Text Chunks",
        len(st.session_state["chunks"])
    )

    col3.metric(
        "FAISS Vectors",
        st.session_state["vector_db"].index.ntotal
    )

    with st.expander("View processed files"):

        for file_name in st.session_state["processed_files"]:
            st.write(f"📄 {file_name}")


# --------------------------------------------------
# 11. QUESTION-ANSWERING INTERFACE
# --------------------------------------------------

if "vector_db" in st.session_state:

    st.divider()

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What is the main idea of Attention Is All You Need?"
    )

    ask_button = st.button(
        "Get Answer",
        type="primary"
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching documents and generating an answer..."
            ):

                try:

                    vector_db = st.session_state["vector_db"]

                    # --------------------------------
                    # SELECT KNOWLEDGE SOURCES
                    # --------------------------------

                    use_papers = response_mode != "General Knowledge Only"
                    use_model_knowledge = response_mode != "Research Papers Only"

                    if use_papers:
                        relevant_docs = vector_db.similarity_search(
                            question,
                            k=min(top_k, vector_db.index.ntotal)
                        )
                    else:
                        relevant_docs = []

                    # --------------------------------
                    # PREPARE PAPER CONTEXT
                    # --------------------------------

                    context_parts = []

                    for doc in relevant_docs:

                        source = doc.metadata.get(
                            "source",
                            "Unknown"
                        )

                        page = doc.metadata.get(
                            "page"
                        )

                        source_label = source

                        if page is not None:
                            source_label += f", page {page}"

                        context_parts.append(
                            f"Source: {source_label}\n"
                            f"Content:\n{doc.page_content}"
                        )

                    context = "\n\n".join(
                        context_parts
                    )

                    # --------------------------------
                    # PROMPT TEMPLATE
                    # --------------------------------

                    prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                """
                                You are a helpful research and general-purpose assistant.

                                Selected answering mode: {mode}

                                Follow the rules for the selected mode:

                                - If the mode is "Research Papers Only":
                                  Answer only from the supplied document context.
                                  Do not fill gaps with pretrained knowledge. If the
                                  context does not contain the answer, say that the
                                  uploaded papers do not provide enough information.

                                - If the mode is "General Knowledge Only":
                                  Answer from your pretrained knowledge. Ignore the
                                  uploaded-paper context for factual claims. Be clear
                                  when something is uncertain or may have changed.

                                - If the mode is "Hybrid (Papers + Model Knowledge)":
                                  Use the document context first for paper-specific
                                  claims, and use pretrained knowledge to explain,
                                  define, or add general background. Clearly distinguish
                                  what the papers say from general background. Never
                                  claim a general-knowledge statement came from a paper
                                  unless the context supports it.

                                In every mode:
                                1. Do not fabricate paper findings, quotations, or citations.
                                2. Explain clearly and use bullet points when helpful.
                                3. Mention relevant source filenames and page numbers
                                   when using supplied paper context.

                                Document context:
                                {context}
                                """
                            ),
                            (
                                "human",
                                "{question}"
                            )
                        ]
                    )

                    # --------------------------------
                    # LOAD LLM
                    # --------------------------------

                    llm = load_llm(
                        ollama_model
                    )

                    # --------------------------------
                    # BUILD RAG CHAIN
                    # --------------------------------

                    output_parser = StrOutputParser()

                    rag_chain = (
                        prompt
                        | llm
                        | output_parser
                    )

                    # --------------------------------
                    # GENERATE ANSWER
                    # --------------------------------

                    answer = rag_chain.invoke(
                        {
                            "context": context if use_papers else "No paper context used in this mode.",
                            "question": question,
                            "mode": response_mode
                        }
                    )

                    st.subheader("📝 Answer")

                    st.markdown(answer)

                    # --------------------------------
                    # SHOW RETRIEVED SOURCES
                    # --------------------------------

                    if use_papers and relevant_docs:
                        with st.expander(
                            "🔎 View relevant source passages"
                        ):

                            for index, doc in enumerate(
                                relevant_docs,
                                start=1
                            ):

                                st.markdown(
                                    f"**Source {index}**"
                                )

                                st.caption(
                                    f"File: {doc.metadata.get('source', 'Unknown')}"
                                    + (
                                        f" | Page: {doc.metadata['page']}"
                                        if "page" in doc.metadata
                                        else ""
                                    )
                                )

                                st.write(doc.page_content)
                                st.divider()

                except Exception as error:

                    st.error(
                        f"Question answering failed: {error}"
                    )

                    st.info(
                        "Check that Ollama is running and the selected "
                        "model is available."
                    )


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit, LangChain, Hugging Face, "
    "FAISS and Ollama."
)