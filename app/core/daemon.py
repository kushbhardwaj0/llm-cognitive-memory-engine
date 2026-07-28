import asyncio
import logging
from typing import Optional

from app.memory.episodic import EpisodicMemory
from app.memory.semantic import SemanticMemory
from app.algorithms.extraction import ConceptExtractor

logger = logging.getLogger(__name__)


class ConsolidationDaemon:
    """
    Asynchronous background task daemon that periodically pulls unconsolidated raw episodes
    from Episodic Memory (SQLite), extracts concept triples using the local LLM, and merges
    them into Semantic Memory (NetworkX Knowledge Graph).
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        concept_extractor: Optional[ConceptExtractor] = None,
        interval_seconds: float = 10.0,
        batch_size: int = 10,
    ):
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.concept_extractor = concept_extractor or ConceptExtractor()
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def consolidate_once(self) -> int:
        """
        Execute a single consolidation pass over un-consolidated episodes.
        Returns the number of episodes consolidated.
        """
        unconsolidated = await self.episodic_memory.get_unconsolidated_episodes(
            limit=self.batch_size
        )
        if not unconsolidated:
            return 0

        logger.info(f"Consolidating {len(unconsolidated)} raw episodic logs into Semantic Memory...")

        # Combine episodic contents for batch LLM extraction
        combined_text = "\n".join([f"{ep.role}: {ep.content}" for ep in unconsolidated])

        # Extract triples using LLM
        triples = await self.concept_extractor.extract_triples(combined_text)

        # Merge extracted triples into Semantic Knowledge Graph
        for t in triples:
            self.semantic_memory.add_triple(
                subject=t.subject,
                predicate=t.predicate,
                obj=t.object,
            )

        # Apply Ebbinghaus decay pruning and save updated graph
        self.semantic_memory.apply_decay()
        self.semantic_memory.save()

        # Mark episodes as consolidated (consolidated = 1) in SQLite
        episode_ids = [ep.id for ep in unconsolidated]
        await self.episodic_memory.mark_consolidated(episode_ids)

        logger.info(
            f"Successfully consolidated {len(unconsolidated)} episodes and added {len(triples)} knowledge triples."
        )
        return len(unconsolidated)

    async def _run_loop(self) -> None:
        """Internal background loop running periodic consolidation cycles."""
        logger.info("ConsolidationDaemon background loop started.")
        while self._running:
            try:
                await self.consolidate_once()
            except Exception as e:
                logger.error(f"Error during background consolidation loop: {e}")

            # Sleep until next consolidation cycle
            await asyncio.sleep(self.interval_seconds)

    def start(self) -> None:
        """Start the background consolidation task loop."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run_loop())
            logger.info("ConsolidationDaemon task scheduled.")

    async def stop(self) -> None:
        """Stop the background consolidation task loop gracefully."""
        if self._running:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            logger.info("ConsolidationDaemon stopped gracefully.")
