"""Persisted session state models for CLI conversations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

type SessionMessage = dict[str, Any]


def _normalize_preview(text: str, *, limit: int = 120) -> str:
    """Collapse whitespace and keep session previews compact."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


class SessionState(BaseModel):
    """Persisted conversation state for a single chat session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    working_dir: Path
    model: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    history: list[SessionMessage] = Field(default_factory=list)
    interaction_log_path: Path | None = None

    @property
    def message_count(self) -> int:
        """Return the number of persisted conversation messages."""
        return len(self.history)

    @property
    def last_user_message(self) -> str | None:
        """Return a short preview of the latest plain-text user prompt."""
        for message in reversed(self.history):
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return _normalize_preview(content)

        return None
