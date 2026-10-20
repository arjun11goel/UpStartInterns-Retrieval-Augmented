"""Retrieve relevant chunks using FAISS distance."""
from langchain_community.vectorstores import FAISS
from rag.config import MAX_L2_DISTANCE, TOP_K
from rag.schemas import RetrievedPassage

def retrieve_passages(vector_store: FAISS, question: str) -> list[RetrievedPassage]:
    results = vector_store.similarity_search_with_score(question, k=TOP_K)
    passages = []
    for doc, distance in results:
        distance = float(distance)
        if distance > MAX_L2_DISTANCE:
            continue
        passages.append(RetrievedPassage(
            text=doc.page_content,
            source=doc.metadata.get("source", "Unknown source"),
            page=doc.metadata.get("page"),
            similarity=1.0 / (1.0 + max(distance, 0.0)),
        ))
    return passages
