import time
import uuid
import math
from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings


def estimate_tokens(text: str) -> int:
    """Fast heuristics for token count estimation (~1.3 tokens per word + 4 overhead tokens)."""
    if not text:
        return 0
    words = text.split()
    return math.ceil(len(words) * 1.3) + 4


class Message(BaseModel):
    """Represents a single message in Working Memory."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = Field(default_factory=time.time)
    token_count: int = 0

    def model_post_init(self, __context):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


class WorkingMemory:
    """In-memory sliding token window buffer for active conversation state."""

    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens or settings.WORKING_MEMORY_MAX_TOKENS
        self._buffer: List[Message] = []

    @property
    def total_tokens(self) -> int:
        """Calculate aggregate token count of active buffer."""
        return sum(msg.token_count for msg in self._buffer)

    def add_message(self, message: Message) -> List[Message]:
        """
        Add a message to the working memory buffer.
        If total tokens exceed max_tokens, pop oldest messages and return them for eviction.
        """
        self._buffer.append(message)
        evicted: List[Message] = []

        # Evict oldest messages while buffer exceeds capacity
        while self.total_tokens > self.max_tokens and len(self._buffer) > 1:
            popped = self._buffer.pop(0)
            evicted.append(popped)

        return evicted

    def get_messages(self) -> List[Message]:
        """Return copy of active messages in working memory buffer."""
        return list(self._buffer)

    def load_messages(self, messages: List[Message]) -> None:
        """Load list of historical messages directly into buffer."""
        self._buffer = list(messages)

    def clear(self) -> None:
        """Clear all active messages from buffer."""
        self._buffer.clear()
