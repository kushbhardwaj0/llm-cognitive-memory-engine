import time
import uuid
import logging
from typing import List, Optional
import aiosqlite
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)


class Episode(BaseModel):
    """Represents a single persistent event log entry in Episodic Memory."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = Field(default_factory=time.time)
    consolidated: bool = False


class EpisodicMemory:
    """Async SQLite persistent event log manager for raw conversation timeline."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.EPISODIC_DB_PATH

    async def initialize(self) -> None:
        """Create database tables and performance indexes if they do not exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    consolidated INTEGER DEFAULT 0
                );
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_session_ts ON episodes(session_id, timestamp);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(session_id, consolidated);"
            )
            await db.commit()
            logger.info(f"EpisodicMemory DB initialized at '{self.db_path}'")

    async def add_episode(self, episode: Episode) -> None:
        """Add a single episode to the persistent store."""
        await self.add_episodes([episode])

    async def add_episodes(self, episodes: List[Episode]) -> None:
        """Batch add multiple episodes to the persistent store."""
        if not episodes:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """
                INSERT INTO episodes (id, session_id, role, content, timestamp, consolidated)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        ep.id,
                        ep.session_id,
                        ep.role,
                        ep.content,
                        ep.timestamp,
                        1 if ep.consolidated else 0,
                    )
                    for ep in episodes
                ],
            )
            await db.commit()

    async def get_recent_episodes(self, session_id: str, limit: int = 10) -> List[Episode]:
        """Fetch the most recent episodes for a session in chronological order."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT id, session_id, role, content, timestamp, consolidated
                FROM episodes
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = await cursor.fetchall()
            episodes = [
                Episode(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    consolidated=bool(row["consolidated"]),
                )
                for row in rows
            ]
            # Return in chronological order (oldest first)
            return episodes[::-1]

    async def get_unconsolidated_episodes(
        self, session_id: Optional[str] = None, limit: int = 50
    ) -> List[Episode]:
        """Fetch unconsolidated episodes (consolidated = 0) for knowledge extraction."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if session_id:
                cursor = await db.execute(
                    """
                    SELECT id, session_id, role, content, timestamp, consolidated
                    FROM episodes
                    WHERE session_id = ? AND consolidated = 0
                    ORDER BY timestamp ASC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT id, session_id, role, content, timestamp, consolidated
                    FROM episodes
                    WHERE consolidated = 0
                    ORDER BY timestamp ASC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = await cursor.fetchall()
            return [
                Episode(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    consolidated=bool(row["consolidated"]),
                )
                for row in rows
            ]

    async def mark_consolidated(self, episode_ids: List[str]) -> None:
        """Mark a list of episode IDs as consolidated = 1."""
        if not episode_ids:
            return

        async with aiosqlite.connect(self.db_path) as db:
            placeholders = ",".join(["?"] * len(episode_ids))
            await db.execute(
                f"UPDATE episodes SET consolidated = 1 WHERE id IN ({placeholders})",
                episode_ids,
            )
            await db.commit()

    async def count_episodes(self, session_id: Optional[str] = None) -> int:
        """Count total episodes stored."""
        async with aiosqlite.connect(self.db_path) as db:
            if session_id:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM episodes WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = await db.execute("SELECT COUNT(*) FROM episodes")
            row = await cursor.fetchone()
            return row[0] if row else 0
