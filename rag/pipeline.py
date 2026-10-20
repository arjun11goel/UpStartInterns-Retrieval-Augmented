"""Orchestrate the complete RAG workflow."""
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rag.chunking import split_documents
from rag.document_loader import extract_uploaded_documents
from rag.generator import FALLBACK, generate_answer
from rag.retriever import retrieve_passages
from rag.schemas import AnswerResult
from rag.vector_store import create_vector_store

@dataclass
class KnowledgeBase:
    vector_store: FAISS
    chunk_count: int

def build_knowledge_base(documents: list[Document]) -> KnowledgeBase:
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("The uploaded files produced no text chunks.")
    return KnowledgeBase(vector_store=create_vector_store(chunks), chunk_count=len(chunks))

def answer_question(knowledge_base: KnowledgeBase, question: str) -> AnswerResult:
    passages = retrieve_passages(knowledge_base.vector_store, question)
    if not passages:
        return AnswerResult(answer=FALLBACK, passages=[], used_fallback=True)
    answer = generate_answer(question, passages)
    return AnswerResult(answer=answer, passages=passages, used_fallback=(answer.strip() == FALLBACK))
