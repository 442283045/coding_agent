"""Tests for persisted session storage helpers."""

import json
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
    assert loaded.name is None
    assert loaded.working_dir == workspace.resolve()
    assert loaded.model == "gpt-4o-mini"
    assert loaded.history == session.history
    assert loaded.message_count == 2
    assert loaded.display_name == "Hello"
    assert loaded.first_user_message == "Hello"
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


def test_session_store_can_delete_saved_sessions(tmp_path: Path) -> None:
    """Deleting a saved session should remove its persisted JSON file."""
    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")

    deleted_path = store.delete_session(session.session_id)

    assert deleted_path.name == f"{session.session_id}.json"
    assert not deleted_path.exists()


def test_session_store_can_rename_saved_sessions(tmp_path: Path) -> None:
    """Renaming a session should persist an explicit display name."""
    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    renamed = store.rename_session(session.session_id, "  Bug triage   sprint  ")

    assert renamed.name == "Bug triage sprint"
    assert renamed.display_name == "Bug triage sprint"
    assert store.load_session(session.session_id).name == "Bug triage sprint"


def test_session_store_can_clear_an_explicit_session_name(tmp_path: Path) -> None:
    """Clearing a session name should fall back to prompt-derived naming."""
    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    session.history.append({"role": "user", "content": "Investigate regressions"})
    store.save_session(session)

    renamed = store.rename_session(session.session_id, "Triage")
    cleared = store.clear_session_name(session.session_id)

    assert renamed.display_name == "Triage"
    assert cleared.name is None
    assert cleared.display_name == "Investigate regressions"


def test_session_display_name_uses_the_first_user_prompt(tmp_path: Path) -> None:
    """Session names should stay anchored to the first user prompt."""
    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    session.history.extend(
        [
            {"role": "user", "content": "First task name"},
            {"role": "assistant", "content": "Working on it"},
            {"role": "user", "content": "A later follow-up prompt"},
        ]
    )
    store.save_session(session)

    loaded = store.load_session(session.session_id)

    assert loaded.display_name == "First task name"
    assert loaded.first_user_message == "First task name"
    assert loaded.last_user_message == "A later follow-up prompt"


def test_session_exports_are_json_and_markdown_friendly(tmp_path: Path) -> None:
    """Session export helpers should support JSON and Markdown consumers."""
    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    session.history.extend(
        [
            {"role": "user", "content": "Summarize this repo"},
            {"role": "assistant", "content": "Here is a summary"},
        ]
    )
    store.save_session(session)
    store.rename_session(session.session_id, "Repository summary")

    json_payload = store.export_session_json(session.session_id)
    markdown = store.export_session_markdown(session.session_id)

    assert json.loads(json.dumps(json_payload)) == json_payload
    assert json_payload["name"] == "Repository summary"
    assert json_payload["display_name"] == "Repository summary"
    assert json_payload["history"][0]["content"] == "Summarize this repo"
    assert "# Session `" in markdown
    assert "Repository summary" in markdown
    assert "## Transcript" in markdown
    assert "### 1. User" in markdown
