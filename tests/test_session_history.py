"""Tests for persisted session storage helpers."""

from pathlib import Path

import pytest

from coding_agent.agent.history import SessionNotFoundError, SessionStore
from coding_agent.config import Settings


def test_session_store_round_trips_saved_sessions(tmp_path: Path) -> None:
    """SessionStore should persist and restore conversation history."""
    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    session.history.extend(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    )
    store.save_session(session)

    loaded = store.load_session(session.session_id)

    assert loaded.session_id == session.session_id
    assert loaded.working_dir == workspace.resolve()
    assert loaded.model == "gpt-4o-mini"
    assert loaded.history == session.history
    assert loaded.message_count == 2
    assert loaded.last_user_message == "Hello"


def test_session_store_lists_newest_sessions_first(tmp_path: Path) -> None:
    """Session listings should be sorted by most recently updated first."""
    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    older = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    newer = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    older.history.append({"role": "user", "content": "old"})
    newer.history.append({"role": "user", "content": "new"})
    store.save_session(older)
    store.save_session(newer)

    sessions = store.list_sessions()

    assert [session.session_id for session in sessions] == [newer.session_id, older.session_id]


def test_session_store_rejects_unknown_session_ids(tmp_path: Path) -> None:
    """Loading an unknown session id should raise a dedicated error."""
    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")

    with pytest.raises(SessionNotFoundError):
        store.load_session("missing-session")
