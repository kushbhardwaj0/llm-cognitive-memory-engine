import math
from typing import List, Tuple
from app.llm.client import BaseLLMClient, get_llm_client


class BaselineVectorRAG:
    """
    Standard Naive RAG Baseline for Token Efficiency Benchmarks.
    """

    def __init__(self, llm_client: BaseLLMClient | None = None, top_k: int = 3):
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

    async def query(self, user_query: str) -> Tuple[str, int]:
        retrieved_chunks = self.retrieve(user_query)
        context_block = "\n---\n".join(retrieved_chunks) if retrieved_chunks else "No context retrieved."

        system_prompt = (
            "You are a helpful AI assistant. Use the raw text context provided below to answer the user question:\n\n"
            f"[RETRIEVED RAG CONTEXT]\n{context_block}"
        )

        full_prompt = f"User Question: {user_query}\nAssistant:"
        words = (system_prompt + "\n" + full_prompt).split()
        estimated_prompt_tokens = math.ceil(len(words) * 1.3)

        response_text = await self.llm_client.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
        )

        return response_text, estimated_prompt_tokens
