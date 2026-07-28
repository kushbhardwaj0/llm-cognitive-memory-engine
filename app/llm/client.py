import json
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Optional
import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract Base Class for LLM Client implementations."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate text response asynchronously."""
        pass

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        """Generate structured JSON response validated against a Pydantic schema."""
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """Verify LLM service availability and readiness."""
        pass


class OllamaClient(BaseLLMClient):
    """Asynchronous Ollama LLM Client using httpx."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    async def check_health(self) -> bool:
        """Check if Ollama server is reachable and respond with HTTP 200."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Send chat request to Ollama /api/chat endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "").strip()
            except httpx.HTTPError as err:
                logger.error(f"HTTP error during LLM generation: {err}")
                raise RuntimeError(f"Ollama request failed: {err}") from err

    async def generate_json(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        """Request JSON formatted output from Ollama and validate with Pydantic."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        json_system_prompt = (
            f"You are a precise data extraction assistant. "
            f"You MUST output raw JSON adhering strictly to this JSON schema:\n{schema_json}\n"
            f"Do not wrap the output in markdown codeblocks or return extra text. Output ONLY valid JSON."
        )
        if system_prompt:
            json_system_prompt = f"{system_prompt}\n\n{json_system_prompt}"

        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                content = response.json().get("message", {}).get("content", "").strip()
                
                # Sanitize response content in case model includes markdown ticks or conversational text
                if content.startswith("```"):
                    lines = content.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    content = "\n".join(lines).strip()

                # Extract JSON substring between outer { } or [ ]
                start_brace = content.find("{")
                start_bracket = content.find("[")
                starts = [i for i in (start_brace, start_bracket) if i != -1]
                if starts:
                    first_idx = min(starts)
                    last_idx = max(content.rfind("}"), content.rfind("]"))
                    if last_idx > first_idx:
                        content = content[first_idx : last_idx + 1]

                return schema.model_validate_json(content)
            except (httpx.HTTPError, ValidationError, json.JSONDecodeError) as err:
                logger.error(f"Failed to generate valid JSON schema: {err}")
                raise RuntimeError(f"JSON Generation & Parsing error: {err}") from err


def get_llm_client() -> BaseLLMClient:
    """Factory function to get default configured LLM client."""
    return OllamaClient()
