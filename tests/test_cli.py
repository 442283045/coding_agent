"""Tests for the CLI entrypoint."""

import json
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

import coding_agent.cli as cli_module
from coding_agent.agent.history import SessionStore
from coding_agent.config import Settings
from coding_agent.doctor import DoctorCheck, DoctorReport

runner = CliRunner()


def test_root_command_starts_interactive_session(monkeypatch, tmp_path: Path) -> None:
    """Running the root command without arguments should start interactive mode."""

    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(
            self,
            working_dir: str | None,
            model: str,
            debug: bool,
            resume_session_id: str | None = None,
        ) -> None:
            captured["working_dir"] = working_dir
            captured["model"] = model
            captured["debug"] = debug
            captured["resume_session_id"] = resume_session_id
            self.working_dir = tmp_path.resolve()
            self.model = model
            self.session_id = resume_session_id or "session-new"

        def run_interactive(self) -> None:
            captured["interactive_started"] = True

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(width=200, force_terminal=False, color_system=None),
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli_module.app, [])

    assert result.exit_code == 0
    assert captured["working_dir"] == str(tmp_path.resolve())
    assert captured["model"] == cli_module.DEFAULT_MODEL
    assert captured["debug"] is cli_module.DEFAULT_DEBUG
    assert captured["resume_session_id"] is None
    assert captured["interactive_started"] is True
    assert "Coding Agent" in result.output
    assert f"Workspace: {tmp_path.resolve()}" in result.output
    assert "Session: session-new" in result.output


def test_root_command_accepts_workspace_option(monkeypatch, tmp_path: Path) -> None:
    """The root interactive command should accept --workspace/-w."""

    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(
            self,
            working_dir: str | None,
            model: str,
            debug: bool,
            resume_session_id: str | None = None,
        ) -> None:
            captured["working_dir"] = working_dir
            captured["model"] = model
            captured["debug"] = debug
            captured["resume_session_id"] = resume_session_id
            self.working_dir = tmp_path.resolve()
            self.model = model
            self.session_id = resume_session_id or "session-new"

        def run_interactive(self) -> None:
            captured["interactive_started"] = True

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(width=200, force_terminal=False, color_system=None),
    )

    result = runner.invoke(cli_module.app, ["-w", str(tmp_path)])

    assert result.exit_code == 0
    assert captured["working_dir"] == str(tmp_path.resolve())
    assert captured["resume_session_id"] is None
    assert captured["interactive_started"] is True
    assert f"Workspace: {tmp_path}" in result.output


def test_chat_banner_uses_ascii_title(monkeypatch, tmp_path: Path) -> None:
    """The chat banner should render safely on non-UTF-8 Windows consoles."""

    class FakeAgent:
        def __init__(
            self,
            working_dir: str | None,
            model: str,
            debug: bool,
            resume_session_id: str | None = None,
        ) -> None:
            self.working_dir = working_dir
            self.model = model
            self.debug = debug
            self.session_id = resume_session_id or "session-new"

        def run_interactive(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(width=200, force_terminal=False, color_system=None),
    )

    result = runner.invoke(cli_module.app, ["chat", str(tmp_path)])

    assert result.exit_code == 0
    assert "Coding Agent" in result.output
    assert f"Workspace: {tmp_path}" in result.output
    assert "🤖 Coding Agent" not in result.output


def test_root_workspace_banner_preserves_relative_workspace_option(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The banner should show the user-provided -w value while resolving the real directory."""

    captured: dict[str, object] = {}
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class FakeAgent:
        def __init__(
            self,
            working_dir: str | None,
            model: str,
            debug: bool,
            resume_session_id: str | None = None,
        ) -> None:
            captured["working_dir"] = working_dir
            captured["resume_session_id"] = resume_session_id
            self.working_dir = workspace.resolve()
            self.model = model
            self.session_id = resume_session_id or "session-new"

        def run_interactive(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(width=200, force_terminal=False, color_system=None),
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli_module.app, ["-w", "workspace"])

    assert result.exit_code == 0
    assert captured["working_dir"] == str(workspace.resolve())
    assert captured["resume_session_id"] is None
    assert "Workspace: workspace" in result.output


def test_run_accepts_workspace_option(monkeypatch, tmp_path: Path) -> None:
    """The run command should accept --workspace/-w as a path alias."""

    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(
            self,
            working_dir: str | None,
            model: str,
            debug: bool,
            resume_session_id: str | None = None,
        ) -> None:
            captured["working_dir"] = working_dir
            captured["model"] = model
            captured["debug"] = debug
            captured["resume_session_id"] = resume_session_id

        def run_once(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "done"

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)

    result = runner.invoke(cli_module.app, ["run", "hello", "-w", str(tmp_path)])

    assert result.exit_code == 0
    assert captured["working_dir"] == str(tmp_path.resolve())
    assert captured["resume_session_id"] is None
    assert captured["prompt"] == "hello"
    assert "done" in result.output


def test_run_renders_markdown_output(monkeypatch, tmp_path: Path) -> None:
    """The run command should render final output as Markdown."""

    rendered_markdown: list[str] = []

    class FakeMarkdown:
        def __init__(self, content: str) -> None:
            self.content = content
            rendered_markdown.append(content)

    class FakeAgent:
        def __init__(
            self,
            working_dir: str | None,
            model: str,
            debug: bool,
            resume_session_id: str | None = None,
        ) -> None:
            return None

        def run_once(self, prompt: str) -> str:
            return "# Title\n\n`code`"

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)
    monkeypatch.setattr(cli_module, "Markdown", FakeMarkdown)

    result = runner.invoke(cli_module.app, ["run", "hello", "-w", str(tmp_path)])

    assert result.exit_code == 0
    assert rendered_markdown == ["# Title\n\n`code`"]


def test_chat_rejects_conflicting_path_and_workspace(
    monkeypatch,
    tmp_path: Path,
    tmp_path_factory,
) -> None:
    """The chat command should reject conflicting path sources."""

    other_path = tmp_path_factory.mktemp("other-workspace")
    monkeypatch.setattr(cli_module, "Agent", object)

    result = runner.invoke(
        cli_module.app,
        ["chat", str(tmp_path), "-w", str(other_path)],
    )

    assert result.exit_code != 0
    assert "Use either the path argument or --workspace/-w, not both." in result.output


def test_chat_accepts_resume_session_option(monkeypatch, tmp_path: Path) -> None:
    """The chat command should resume a saved session without a workspace path."""

    captured: dict[str, object] = {}
    session_workspace = tmp_path / "workspace"
    session_workspace.mkdir()

    class FakeAgent:
        def __init__(
            self,
            working_dir: str | None,
            model: str | None,
            debug: bool,
            resume_session_id: str | None = None,
        ) -> None:
            captured["working_dir"] = working_dir
            captured["model"] = model
            captured["resume_session_id"] = resume_session_id
            self.working_dir = session_workspace.resolve()
            self.model = model or "saved-model"
            self.session_id = resume_session_id or "session-new"

        def run_interactive(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(width=200, force_terminal=False, color_system=None),
    )

    result = runner.invoke(cli_module.app, ["chat", "--resume", "session-123"])

    assert result.exit_code == 0
    assert captured["working_dir"] is None
    assert captured["model"] is None
    assert captured["resume_session_id"] == "session-123"
    assert f"Workspace: {session_workspace.resolve()}" in result.output
    assert "Session: session-123" in result.output


def test_chat_rejects_resume_and_workspace(monkeypatch, tmp_path: Path) -> None:
    """The chat command should reject mixing resume ids with workspace inputs."""

    monkeypatch.setattr(cli_module, "Agent", object)

    result = runner.invoke(
        cli_module.app,
        ["chat", str(tmp_path), "--resume", "session-123"],
    )

    assert result.exit_code != 0
    assert "Use either a workspace path or --resume/-r, not both." in result.output


def test_run_accepts_resume_session_option(monkeypatch) -> None:
    """The run command should support continuing a saved session."""

    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(
            self,
            working_dir: str | None,
            model: str | None,
            debug: bool,
            resume_session_id: str | None = None,
        ) -> None:
            captured["working_dir"] = working_dir
            captured["model"] = model
            captured["resume_session_id"] = resume_session_id

        def run_once(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "continued"

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)

    result = runner.invoke(cli_module.app, ["run", "--resume", "session-123", "continue"])

    assert result.exit_code == 0
    assert captured["working_dir"] is None
    assert captured["model"] is None
    assert captured["resume_session_id"] == "session-123"
    assert captured["prompt"] == "continue"
    assert "continued" in result.output


def test_sessions_list_renders_saved_sessions(monkeypatch, tmp_path: Path) -> None:
    """The sessions list command should show persisted session summaries."""

    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    session.history.append({"role": "user", "content": "Investigate bug"})
    store.save_session(session)

    monkeypatch.setattr(cli_module, "_get_session_store", lambda: store)
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(width=240, force_terminal=False, color_system=None),
    )

    result = runner.invoke(cli_module.app, ["sessions", "list"])

    assert result.exit_code == 0
    assert "Saved Sessions" in result.output
    assert session.session_id in result.output
    assert "Investigate bug" in result.output
    assert "Last User Message" not in result.output


def test_sessions_show_renders_resume_hint(monkeypatch, tmp_path: Path) -> None:
    """The sessions show command should render session metadata and naming."""

    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    session.history.append({"role": "user", "content": "Review the parser"})
    store.save_session(session)

    monkeypatch.setattr(cli_module, "_get_session_store", lambda: store)
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(width=240, force_terminal=False, color_system=None),
    )

    result = runner.invoke(cli_module.app, ["sessions", "show", session.session_id])

    assert result.exit_code == 0
    assert session.session_id in result.output
    assert "Review the parser" in result.output
    assert "Name:" in result.output
    assert "Resume with:" in result.output
    assert f"coding-agent chat --resume {session.session_id}" in result.output


def test_sessions_rename_updates_explicit_name(monkeypatch, tmp_path: Path) -> None:
    """The sessions rename command should persist a custom session name."""

    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")

    monkeypatch.setattr(cli_module, "_get_session_store", lambda: store)

    result = runner.invoke(
        cli_module.app,
        ["sessions", "rename", session.session_id, "Parser triage"],
    )

    assert result.exit_code == 0
    assert "Parser triage" in result.output
    assert store.load_session(session.session_id).name == "Parser triage"


def test_sessions_rename_can_clear_explicit_name(monkeypatch, tmp_path: Path) -> None:
    """The sessions rename command should clear explicit names on request."""

    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    session.history.append({"role": "user", "content": "Investigate failing tests"})
    store.save_session(session)
    store.rename_session(session.session_id, "Failing tests")

    monkeypatch.setattr(cli_module, "_get_session_store", lambda: store)

    result = runner.invoke(
        cli_module.app,
        ["sessions", "rename", session.session_id, "--clear"],
    )

    assert result.exit_code == 0
    assert "Investigate failing tests" in result.output
    assert store.load_session(session.session_id).name is None


def test_sessions_export_writes_json_output(monkeypatch, tmp_path: Path) -> None:
    """The sessions export command should write JSON exports to disk."""

    store = SessionStore(Settings(_env_file=None), sessions_dir=tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = store.create_session(working_dir=workspace, model="gpt-4o-mini")
    session.history.append({"role": "user", "content": "Summarize this repo"})
    store.save_session(session)
    store.rename_session(session.session_id, "Repository summary")
    output_path = tmp_path / "exports" / "session.json"

    monkeypatch.setattr(cli_module, "_get_session_store", lambda: store)

    result = runner.invoke(
        cli_module.app,
        [
            "sessions",
            "export",
            session.session_id,
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
    )

    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert output_path.exists()
    assert exported["display_name"] == "Repository summary"
    assert exported["history"][0]["content"] == "Summarize this repo"


def test_doctor_command_renders_report_and_exits_on_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The doctor command should render checks and exit non-zero on blocking issues."""

    report = DoctorReport(
        working_dir=tmp_path.resolve(),
        checks=(
            DoctorCheck(
                key="default_model",
                label="Default model",
                status="ok",
                summary="Configured default model is `gpt-4o-mini`.",
            ),
            DoctorCheck(
                key="provider_credentials",
                label="Provider credentials",
                status="fail",
                summary="`gpt-4o-mini` requires `OPENAI_API_KEY`, but it is not configured.",
            ),
        ),
    )

    monkeypatch.setattr(cli_module, "build_doctor_report", lambda settings_obj, working_dir: report)
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(width=240, force_terminal=False, color_system=None),
    )

    result = runner.invoke(cli_module.app, ["doctor", "-w", str(tmp_path)])

    assert result.exit_code == 1
    assert "Coding Agent Doctor FAIL" in result.output
    assert "Default model" in result.output
    assert "Provider credentials" in result.output
