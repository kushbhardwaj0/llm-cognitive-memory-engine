import asyncio
import sys
from pydantic import BaseModel, Field
from app.core.config import settings
from app.llm.client import OllamaClient


class ExtractedConcept(BaseModel):
    subject: str = Field(description="The primary entity or subject")
    relation: str = Field(description="The relationship or action")
    object: str = Field(description="The target entity or object")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


async def main():
    print("=" * 60)
    print(" Step 1 Verification: Local Ollama & Environment Setup")
    print("=" * 60)

    print(f"[1] Configuration Loaded:")
    print(f"    - Ollama Base URL: {settings.OLLAMA_BASE_URL}")
    print(f"    - Ollama Model:    {settings.OLLAMA_MODEL}")
    print(f"    - Timeout:         {settings.OLLAMA_TIMEOUT}s")
    print("-" * 60)

    client = OllamaClient()

    # Test 1: Health Check
    print("[2] Testing Ollama Service Health...")
    is_healthy = await client.check_health()
    if not is_healthy:
        print("[FAIL] Ollama service is unreachable at", settings.OLLAMA_BASE_URL)
        print("   Please make sure Ollama is running (`ollama serve`) and model exists.")
        sys.exit(1)
    print("[SUCCESS] Ollama daemon is reachable.")
    print("-" * 60)

    # Test 2: Freeform Text Generation
    print(f"[3] Testing Text Generation with model '{settings.OLLAMA_MODEL}'...")
    prompt = "Explain in 2 sentences what makes asynchronous I/O useful for local LLM pipelines."
    try:
        response = await client.generate(prompt=prompt, temperature=0.7)
        print(f"[SUCCESS] Response received:\n\"{response}\"")
    except Exception as e:
        print(f"[FAIL] Text generation error: {e}")
        sys.exit(1)
    print("-" * 60)

    # Test 3: Structured JSON Generation & Validation
    print("[4] Testing Structured JSON Schema Generation & Pydantic Validation...")
    extract_prompt = "Extract key relationship: 'Alice is building a Cognitive Memory Engine in Python.'"
    try:
        extracted: ExtractedConcept = await client.generate_json(
            prompt=extract_prompt,
            schema=ExtractedConcept
        )
        print(f"[SUCCESS] Parsed Pydantic Object successfully:")
        print(f"    - Subject:    {extracted.subject}")
        print(f"    - Relation:   {extracted.relation}")
        print(f"    - Object:     {extracted.object}")
        print(f"    - Confidence: {extracted.confidence}")
    except Exception as e:
        print(f"[FAIL] JSON schema extraction failed: {e}")
        sys.exit(1)

    print("=" * 60)
    print(" ALL STEP 1 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
