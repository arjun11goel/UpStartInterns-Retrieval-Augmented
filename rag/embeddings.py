"""Local Sentence Transformers embeddings."""
from langchain_huggingface import HuggingFaceEmbeddings
from rag.config import EMBEDDING_MODEL

def create_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
