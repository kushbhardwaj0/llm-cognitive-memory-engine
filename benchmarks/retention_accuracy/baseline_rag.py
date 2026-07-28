import math
from typing import List, Tuple
from app.llm.client import BaseLLMClient, get_llm_client


class BaselineVectorRAG:
    """
    Naive RAG baseline for retention stress testing.
    Retrieves Top-K raw text chunks using keyword/vector similarity.
    """

    def __init__(self, llm_client: BaseLLMClient | None = None, top_k: int = 2):
        self.llm_client = llm_client or get_llm_client()
        self.top_k = top_k
        self.documents: List[str] = []

    def add_document(self, text: str) -> None:
        if text and text.strip():
            self.documents.append(text.strip())

    def _calculate_similarity(self, query: str, doc: str) -> float:
        query_words = set(query.lower().split())
        doc_words = set(doc.lower().split())
        if not query_words or not doc_words:
            return 0.0
        intersection = query_words.intersection(doc_words)
        return len(intersection) / math.sqrt(len(query_words) * len(doc_words))

    def retrieve(self, query: str) -> List[str]:
        if not self.documents:
            return []
        scored_docs: List[Tuple[float, str]] = [
            (self._calculate_similarity(query, doc), doc) for doc in self.documents
        ]
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[: self.top_k]]

    async def query(self, user_query: str) -> str:
        retrieved_chunks = self.retrieve(user_query)
        context_block = "\n---\n".join(retrieved_chunks) if retrieved_chunks else "No context retrieved."

        system_prompt = (
            "You are a helpful AI assistant. Use ONLY the raw context below to answer the user question:\n\n"
            f"[RETRIEVED CONTEXT]\n{context_block}"
        )

        full_prompt = f"Question: {user_query}\nAnswer:"

        return await self.llm_client.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )
