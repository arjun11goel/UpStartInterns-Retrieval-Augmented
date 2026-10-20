"""Local FAISS vector index."""
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rag.embeddings import create_embedding_model

def create_vector_store(chunks: list[Document]) -> FAISS:
    if not chunks:
        raise ValueError("No document chunks are available.")
    return FAISS.from_documents(chunks, create_embedding_model())
