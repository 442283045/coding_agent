"""Helpers for saving, loading, and listing persisted chat sessions."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from secrets import token_hex

from pydantic import ValidationError

from coding_agent.agent.state import SessionState
from coding_agent.config import Settings

_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SessionStoreError(RuntimeError):
    """Base error raised for persisted session storage failures."""


class SessionNotFoundError(SessionStoreError):
    """Raised when a requested session id does not exist."""


class SessionStore:
    """Persist and restore agent conversation sessions on disk."""

    def __init__(
        self,
        settings_obj: Settings,
        *,
        sessions_dir: Path | None = None,
    ) -> None:
        self._settings = settings_obj
        self._sessions_dir = (
            sessions_dir.resolve()
            if sessions_dir is not None
            else (settings_obj.config_dir / "sessions").resolve()
        )

    @property
    def sessions_dir(self) -> Path:
        """Return the directory that stores session JSON files."""
        return self._sessions_dir

    def ensure_sessions_dir(self) -> Path:
        """Create the session storage directory when needed."""
        self._settings.ensure_config_dir()
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        return self._sessions_dir

    def create_session(
        self,
        *,
        working_dir: Path,
        model: str,
    ) -> SessionState:
        """Create and persist a new empty session."""
        now = datetime.now().astimezone()
        session = SessionState(
            session_id=self._generate_session_id(now),
            working_dir=working_dir.resolve(),
            model=model,
            created_at=now,
            updated_at=now,
        )
        self.save_session(session)
        return session

    def save_session(self, session: SessionState) -> Path:
        """Write a session snapshot to disk."""
        self.ensure_sessions_dir()
        session.updated_at = datetime.now().astimezone()
        path = self.session_path(session.session_id)
        path.write_text(
            session.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def load_session(self, session_id: str) -> SessionState:
        """Load a previously saved session by id."""
        path = self.session_path(session_id)
        if not path.is_file():
            raise SessionNotFoundError(f"Session `{session_id}` was not found.")

        try:
            return SessionState.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise SessionStoreError(f"Session `{session_id}` is invalid: {exc}") from exc

    def list_sessions(self) -> list[SessionState]:
        """Return all saved sessions, newest first."""
        if not self.sessions_dir.exists():
            return []

        sessions: list[SessionState] = []
        for path in sorted(self.sessions_dir.glob("*.json")):
            try:
                sessions.append(SessionState.model_validate_json(path.read_text(encoding="utf-8")))
            except ValidationError as exc:
                raise SessionStoreError(f"Session file `{path}` is invalid: {exc}") from exc

        return sorted(sessions, key=lambda session: session.updated_at, reverse=True)

    def session_path(self, session_id: str) -> Path:
        """Return the on-disk path for a session id."""
        normalized_id = self._validate_session_id(session_id)
        return self.sessions_dir / f"{normalized_id}.json"

    def _generate_session_id(self, now: datetime) -> str:
        """Generate a readable session id."""
        return f"{now.strftime('%Y%m%d-%H%M%S')}-{token_hex(4)}"

    def _validate_session_id(self, session_id: str) -> str:
        """Reject invalid session ids before touching the filesystem."""
        normalized_id = session_id.strip()
        if not normalized_id or not _SESSION_ID_PATTERN.fullmatch(normalized_id):
            raise SessionStoreError(
                "Session ids may only contain letters, numbers, dots, underscores, and dashes."
            )
        return normalized_id
