import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.core.engine import CognitiveMemoryEngine

router = APIRouter()

# Global Engine Singleton (initialized in main.py lifespan)
engine_instance: Optional[CognitiveMemoryEngine] = None


def get_engine() -> CognitiveMemoryEngine:
    if engine_instance is None:
        raise HTTPException(status_code=500, detail="CognitiveMemoryEngine is not initialized.")
    return engine_instance


class ChatRequest(BaseModel):
    session_id: str = Field(default="default-session", description="Session identifier")
    message: str = Field(..., description="User message prompt")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    working_memory_tokens: int


class MemoryStateResponse(BaseModel):
    working_memory_count: int
    working_memory_tokens: int
    episodic_episodes_count: int
    semantic_nodes_count: int
    semantic_edges_count: int
    semantic_triples: List[Dict[str, str]]


class ConsolidateResponse(BaseModel):
    consolidated_count: int
    semantic_nodes_count: int
    semantic_edges_count: int


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    engine: CognitiveMemoryEngine = Depends(get_engine),
):
    """Execute a chat query through the Cognitive Memory Engine."""
    try:
        response_text = await engine.chat(session_id=req.session_id, user_message=req.message)
        return ChatResponse(
            session_id=req.session_id,
            response=response_text,
            working_memory_tokens=engine.working_memory.total_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine chat error: {str(e)}")


@router.get("/memory/state", response_model=MemoryStateResponse)
async def get_memory_state(
    session_id: Optional[str] = None,
    engine: CognitiveMemoryEngine = Depends(get_engine),
):
    """Inspect current state across Working Memory, Episodic SQLite, and Semantic Graph."""
    working_count = len(engine.working_memory.get_messages())
    working_tokens = engine.working_memory.total_tokens
    episodic_count = await engine.episodic_memory.count_episodes(session_id=session_id)

    graph = engine.semantic_memory.graph
    nodes_count = len(graph.nodes)
    edges_count = len(graph.edges)

    triples = []
    for u, v, data in graph.edges(data=True):
        triples.append({
            "subject": u,
            "predicate": data.get("relation", "RELATED_TO"),
            "object": v,
        })

    return MemoryStateResponse(
        working_memory_count=working_count,
        working_memory_tokens=working_tokens,
        episodic_episodes_count=episodic_count,
        semantic_nodes_count=nodes_count,
        semantic_edges_count=edges_count,
        semantic_triples=triples,
    )


@router.post("/memory/consolidate", response_model=ConsolidateResponse)
async def trigger_consolidation(
    engine: CognitiveMemoryEngine = Depends(get_engine),
):
    """Trigger a manual consolidation pass over un-consolidated SQLite episodes."""
    try:
        count = await engine.daemon.consolidate_once()
        graph = engine.semantic_memory.graph
        return ConsolidateResponse(
            consolidated_count=count,
            semantic_nodes_count=len(graph.nodes),
            semantic_edges_count=len(graph.edges),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consolidation error: {str(e)}")
