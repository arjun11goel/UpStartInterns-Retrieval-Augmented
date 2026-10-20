import streamlit as st

from rag.config import APP_TITLE, CHAT_MODEL
from rag.pipeline import answer_question, build_knowledge_base, extract_uploaded_documents

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
st.title("📚 LocalDoc QA")
st.caption("Document processing, embeddings, and vector search run locally; Gemini generates answers from retrieved passages.")

with st.sidebar:
    st.header("Configuration")
    st.write(f"**Gemini model:** `{CHAT_MODEL}`")
    st.write("**Embeddings:** local Sentence Transformers")
    st.write("**Vector search:** local FAISS")
    st.caption("A Google Gemini API key is required for answer generation.")
    st.warning("Avoid uploading confidential documents because retrieved passages are sent to Gemini.")

uploads = st.file_uploader("Upload PDF, TXT, or Markdown documents", type=["pdf", "txt", "md"], accept_multiple_files=True)

if st.button("Process documents", type="primary", disabled=not uploads):
    with st.spinner("Reading, chunking, embedding, and indexing documents locally..."):
        try:
            documents, skipped = extract_uploaded_documents(uploads)
            if not documents:
                st.session_state.pop("knowledge_base", None)
                st.error("No readable text found. Use a text-based PDF, TXT, or Markdown file. Scanned PDFs need OCR.")
            else:
                kb = build_knowledge_base(documents)
                st.session_state["knowledge_base"] = kb
                st.session_state["indexed_files"] = [f.name for f in uploads]
                st.session_state["chunk_count"] = kb.chunk_count
                st.session_state["skipped_files"] = skipped
                st.success(f"Processed {len(documents)} page/section(s) into {kb.chunk_count} chunks.")
        except Exception as exc:
            st.error(f"Could not process documents: {exc}")

if st.session_state.get("knowledge_base") is not None:
    kb = st.session_state["knowledge_base"]
    st.success(f"Knowledge base ready: {len(st.session_state.get('indexed_files', []))} file(s), {kb.chunk_count} chunks.")
    with st.expander("Indexed files"):
        for name in st.session_state.get("indexed_files", []):
            st.write(f"- {name}")
        if st.session_state.get("skipped_files"):
            st.warning("Files with no extracted text: " + ", ".join(st.session_state["skipped_files"]))

    question = st.text_input("Ask a question about your documents", placeholder="What are the stages of a RAG pipeline?")
    if st.button("Get answer", disabled=not question.strip()):
        with st.spinner("Retrieving evidence and asking Gemini..."):
            try:
                result = answer_question(kb, question)
                st.subheader("Answer")
                st.write(result.answer)
                if result.used_fallback:
                    st.info("The retrieved evidence was insufficient to generate an answer.")
                st.subheader("Retrieved passages")
                if not result.passages:
                    st.write("No passages passed the retrieval threshold.")
                for i, passage in enumerate(result.passages, start=1):
                    page = f" · Page {passage.page}" if passage.page is not None else ""
                    with st.expander(f"[{i}] {passage.source}{page} · similarity {passage.similarity:.3f}"):
                        st.write(passage.text)
            except Exception as exc:
                st.error(f"Could not answer the question: {exc}")
                st.caption("Check GOOGLE_API_KEY in .env, model availability, and API quota.")

st.divider()
st.caption("Similarity thresholds are approximate. Verify important answers against the passages shown.")
