"""Persisted session state models for CLI conversations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    name: str | None = None
    working_dir: Path
    model: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    history: list[SessionMessage] = Field(default_factory=list)
    interaction_log_path: Path | None = None

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str | None) -> str | None:
        """Normalize explicit session names and reject blank values."""
        if value is None:
            return None

        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Session name cannot be empty.")
        return normalized

    @property
    def message_count(self) -> int:
        """Return the number of persisted conversation messages."""
        return len(self.history)

    @property
    def first_user_message(self) -> str | None:
        """Return a short preview of the first plain-text user prompt."""
        for message in self.history:
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return _normalize_preview(content, limit=60)

        return None

    @property
    def display_name(self) -> str:
        """Return the user-facing session name."""
        if self.name is not None:
            return self.name
        return self.first_user_message or "Untitled session"

    @property
    def last_user_message(self) -> str | None:
        """Return a short preview of the latest plain-text user prompt."""
        for message in reversed(self.history):
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return _normalize_preview(content, limit=80)

        return None

    def to_json_export(self) -> dict[str, Any]:
        """Return a JSON-friendly export payload for the full session."""
        payload = cast(dict[str, Any], json.loads(self.model_dump_json()))
        payload.update(
            {
                "display_name": self.display_name,
                "message_count": self.message_count,
                "first_user_message": self.first_user_message,
                "last_user_message": self.last_user_message,
            }
        )
        return payload

    def to_markdown_export(self) -> str:
        """Render the session as a human-readable Markdown transcript."""
        lines = [
            f"# Session `{self.session_id}`",
            "",
            f"- Name: {self.display_name}",
            f"- Workspace: `{self.working_dir}`",
            f"- Model: `{self.model}`",
            f"- Created: `{self.created_at.isoformat()}`",
            f"- Updated: `{self.updated_at.isoformat()}`",
            f"- Messages: `{self.message_count}`",
        ]

        if self.interaction_log_path is not None:
            lines.append(f"- LLM log: `{self.interaction_log_path}`")

        lines.extend(["", "## Transcript"])
        if not self.history:
            lines.append("No messages yet.")
            return "\n".join(lines)

        for index, message in enumerate(self.history, start=1):
            role = str(message.get("role") or "message").strip() or "message"
            lines.extend(
                [
                    "",
                    f"### {index}. {role.title()}",
                    "",
                    "```json",
                    json.dumps(message, ensure_ascii=False, indent=2, default=str),
                    "```",
                ]
            )

        return "\n".join(lines)
