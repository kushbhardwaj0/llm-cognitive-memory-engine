import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from app.llm.client import BaseLLMClient, get_llm_client

logger = logging.getLogger(__name__)


class ExtractedTriple(BaseModel):
    """Subject-Predicate-Object concept triple."""

    subject: str = Field(description="The primary entity, subject, or concept (e.g. Kush, Python)")
    predicate: str = Field(description="The relationship or action in UPPERCASE (e.g. PREFERS, BUILDS, LIVES_IN)")
    object: str = Field(description="The target entity, object, or property (e.g. Qwen2.5, macOS)")


class ExtractionResult(BaseModel):
    """Container schema for batch extracted concept triples."""

    triples: List[ExtractedTriple] = Field(default_factory=list)


class ConceptExtractor:
    """Uses LLM to extract structured (subject, predicate, object) knowledge triples from text."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or get_llm_client()

    async def extract_triples(self, text: str) -> List[ExtractedTriple]:
        """Analyze text input and extract semantic knowledge triples."""
        if not text or not text.strip():
            return []

        system_prompt = (
            "You are a Knowledge Graph Extraction Engine. "
            "Your task is to extract clear, factual entity-relation triples (subject, predicate, object) "
            "from the provided conversation history. "
            "Keep subjects and objects short, specific entity names. "
            "Use clear uppercase relationship predicates like PREFERS, BUILDS, USES, WORKS_ON, LIVES_IN."
        )

        prompt = f"Extract all key knowledge triples from the following text:\n\n{text}"

        try:
            result: ExtractionResult = await self.llm_client.generate_json(
                prompt=prompt,
                schema=ExtractionResult,
                system_prompt=system_prompt,
                temperature=0.1,
            )
            return result.triples
        except Exception as e:
            logger.error(f"Error during triple extraction: {e}")
            return []
