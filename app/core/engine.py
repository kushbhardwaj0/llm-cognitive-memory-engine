import logging
from typing import Dict, List, Optional

from app.core.config import settings
from app.llm.client import BaseLLMClient, get_llm_client
from app.memory.working import WorkingMemory, Message
from app.memory.episodic import EpisodicMemory, Episode
from app.memory.semantic import SemanticMemory
from app.algorithms.activation import SpreadingActivation
from app.core.daemon import ConsolidationDaemon

logger = logging.getLogger(__name__)


class CognitiveMemoryEngine:
    """
    Unified Orchestrator for the Cognitive Memory Architecture.
    Combines Working Memory, Episodic Memory (SQLite), Semantic Memory (Graph),
    Spreading Activation Retrieval, and Background Consolidation Daemon.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        db_path: Optional[str] = None,
        graph_path: Optional[str] = None,
        working_max_tokens: Optional[int] = None,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.working_memory = WorkingMemory(max_tokens=working_max_tokens)
        self.episodic_memory = EpisodicMemory(db_path=db_path)
        self.semantic_memory = SemanticMemory(graph_path=graph_path)
        self.spreading_activation = SpreadingActivation(semantic_memory=self.semantic_memory)
        self.daemon = ConsolidationDaemon(
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
        )

    async def initialize(self) -> None:
        """Initialize underlying stores and load graph data."""
        await self.episodic_memory.initialize()
        self.semantic_memory.load()
        logger.info("CognitiveMemoryEngine initialized successfully.")

    def start_daemon(self) -> None:
        """Start the background consolidation daemon."""
        self.daemon.start()

    async def stop_daemon(self) -> None:
        """Stop the background consolidation daemon."""
        await self.daemon.stop()

    async def chat(self, session_id: str, user_message: str) -> str:
        """
        Execute a complete chat query turn through the cognitive memory pipeline.
        """
        # Step 1: Retrieve context subgraph via Spreading Activation
        retrieved_triples = self.spreading_activation.get_activated_triples(query=user_message)
        
        graph_context = ""
        if retrieved_triples:
            graph_lines = [f"- {sub} {pred} {obj}" for sub, pred, obj in retrieved_triples]
            graph_context = "\n".join(graph_lines)
        else:
            graph_context = "No specific long-term graph facts retrieved for this query."

        # Step 2: Build System Prompt augmented with retrieved Semantic Graph Memory
        system_prompt = (
            "You are an AI assistant powered by a local Cognitive Memory Engine.\n"
            "Below are relevant facts retrieved from your long-term Semantic Knowledge Graph:\n"
            f"[LONG-TERM SEMANTIC MEMORY]\n{graph_context}\n\n"
            "Use these long-term memory facts whenever relevant to personalize your response."
        )

        # Step 3: Format active conversation history from Working Memory
        active_messages = self.working_memory.get_messages()
        conversation_prompt = ""
        if active_messages:
            lines = [f"{msg.role.capitalize()}: {msg.content}" for msg in active_messages]
            conversation_prompt = "\n".join(lines) + f"\nUser: {user_message}\nAssistant:"
        else:
            conversation_prompt = user_message

        # Step 4: Call LLM
        response_text = await self.llm_client.generate(
            prompt=conversation_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
        )

        # Step 5: Update Working Memory
        user_msg = Message(session_id=session_id, role="user", content=user_message)
        assistant_msg = Message(session_id=session_id, role="assistant", content=response_text)

        self.working_memory.add_message(user_msg)
        self.working_memory.add_message(assistant_msg)

        # Step 6: Log raw turns into Episodic Memory SQLite store
        episodes = [
            Episode(
                id=user_msg.id,
                session_id=user_msg.session_id,
                role=user_msg.role,
                content=user_msg.content,
                timestamp=user_msg.timestamp,
                consolidated=False,
            ),
            Episode(
                id=assistant_msg.id,
                session_id=assistant_msg.session_id,
                role=assistant_msg.role,
                content=assistant_msg.content,
                timestamp=assistant_msg.timestamp,
                consolidated=False,
            ),
        ]
        await self.episodic_memory.add_episodes(episodes)

        return response_text
