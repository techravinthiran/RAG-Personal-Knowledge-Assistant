import os
import streamlit as st
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dataset directory 
DATASET_DIR = Path("dataset")
DATASET_DIR.mkdir(exist_ok=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# RAG Pipeline
@st.cache_resource(show_spinner=False)
def load_rag_pipeline(file_paths: tuple):
    from langchain_community.document_loaders import (
        PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    loaders_map = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".docx": Docx2txtLoader,
        ".csv":  CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls":  UnstructuredExcelLoader,
    }

    all_docs = []
    for fp in file_paths:
        ext = Path(fp).suffix.lower()
        loader_cls = loaders_map.get(ext)
        if loader_cls:
            try:
                all_docs.extend(loader_cls(fp).load())
            except Exception as e:
                st.warning(f"Could not load {Path(fp).name}: {e}")

    if not all_docs:
        return None, 0

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    ).split_documents(all_docs)

    if not chunks:
        return None, 0

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    retriever = FAISS.from_documents(chunks, embeddings).as_retriever(
        search_kwargs={"k": 4}
    )
    return retriever, len(chunks)


def get_llm_response(query: str, context_docs, chat_history: list, model_choice: str) -> dict:
    context = "\n\n---\n\n".join([d.page_content for d in context_docs])
    sources = list({d.metadata.get("source", "Unknown") for d in context_docs})

    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a precise, helpful AI Knowledge Assistant. Answer questions ONLY using the provided context.
If the answer is not in the context, say "I couldn't find relevant information in the uploaded documents."

CONTEXT:
{context}

CHAT HISTORY:
{history_text}

USER QUESTION: {query}

ANSWER:"""

    if model_choice == "Groq (LLaMA3)" and os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return {"answer": resp.choices[0].message.content, "sources": sources}
        except Exception as e:
            st.warning(f"Groq error: {e}. Falling back to HuggingFace.")

    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
        tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
        
        def generate_text_t5(prompt: str, max_length: int = 256) -> str:
            inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        short_prompt = f"Context: {context[:1500]}\n\nQuestion: {query}\n\nAnswer:"
        result = generate_text_t5(short_prompt, max_length=256)
        return {"answer": result, "sources": sources}
    except Exception as e:
        return {"answer": f"LLM unavailable: {e}", "sources": sources}

#  SIDEBAR
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.divider()

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported: PDF, TXT, DOCX, CSV, XLSX",
        type=["pdf", "txt", "docx", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        saved_paths = []
        for uf in uploaded_files:
            dest = DATASET_DIR / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(str(dest))
        st.session_state.processed_files = saved_paths

    if st.button("Build Knowledge Base", use_container_width=True, type="primary"):
        if not st.session_state.processed_files:
            st.error("Please upload at least one document first.")
        else:
            with st.spinner("Processing documents..."):
                load_rag_pipeline.clear()
                retriever, n_chunks = load_rag_pipeline(
                    tuple(st.session_state.processed_files)
                )
                st.session_state.chunk_count = n_chunks
                if retriever:
                    st.success(f"{n_chunks} chunks indexed successfully.")
                else:
                    st.error("Failed to index documents.")

    # Auto-configure Groq as default LLM
    try:
        from config import GROQ_API_KEY
        if GROQ_API_KEY:
            os.environ["GROQ_API_KEY"] = GROQ_API_KEY
            model_choice = "Groq (LLaMA3)"
        else:
            model_choice = "HuggingFace (Flan-T5)"
    except ImportError:
        model_choice = "HuggingFace (Flan-T5)"
    
    # Set default top_k
    top_k = 4

    st.divider()

    st.subheader("Stats")
    col1, col2 = st.columns(2)
    col1.metric("Documents", len(st.session_state.processed_files))
    col2.metric("Chunks", st.session_state.chunk_count)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("In Dataset", len(list(DATASET_DIR.glob("*"))))

    st.divider()

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

#  MAIN AREA
st.title("Personal AI Knowledge Assistant")
st.caption("RAG-powered document Q&A  —  Upload → Index → Ask")
st.divider()

if not st.session_state.messages:
    st.info(
        "**Getting Started**\n\n"
        "1. Upload your documents in the sidebar (PDF, TXT, DOCX, CSV, XLSX)\n"
        "2. Click **Build Knowledge Base** to index them\n"
        "3. Ask any question about your documents below"
    )

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div style="display: flex; justify-content: flex-end; margin: 10px 0;">'
                  f'<div style="background-color: #e1f5fe; padding: 10px; border-radius: 10px; max-width: 70%; text-align: right;">'
                  f'{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="display: flex; justify-content: flex-start; margin: 10px 0;">'
                  f'<div style="background-color: #f5f5f5; padding: 10px; border-radius: 10px; max-width: 70%;">'
                  f'{msg["content"]}</div></div>', unsafe_allow_html=True)
        if "sources" in msg and msg["sources"]:
            st.caption("Sources: " + ", ".join(Path(s).name for s in msg["sources"]))

# Chat input
if query := st.chat_input("Ask anything about your documents..."):
    if not st.session_state.processed_files or st.session_state.chunk_count == 0:
        st.warning("Please upload documents and click **Build Knowledge Base** first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div style="display: flex; justify-content: flex-end; margin: 10px 0;">'
              f'<div style="background-color: #e1f5fe; padding: 10px; border-radius: 10px; max-width: 70%; text-align: right;">'
              f'{query}</div></div>', unsafe_allow_html=True)

    with st.spinner("Searching documents..."):
        retriever, _ = load_rag_pipeline(tuple(st.session_state.processed_files))
        if retriever is None:
            st.error("Knowledge base not ready.")
            st.stop()
        retriever.search_kwargs["k"] = top_k
        relevant_docs = retriever.invoke(query)

    with st.spinner("Generating answer..."):
        result = get_llm_response(query, relevant_docs, st.session_state.messages, model_choice)

    st.markdown(f'<div style="display: flex; justify-content: flex-start; margin: 10px 0;">'
              f'<div style="background-color: #f5f5f5; padding: 10px; border-radius: 10px; max-width: 70%;">'
              f'{result["answer"]}</div></div>', unsafe_allow_html=True)
    if result["sources"]:
        st.caption("Sources: " + ", ".join(Path(s).name for s in result["sources"]))

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
    st.session_state.total_queries += 1
    st.rerun()
