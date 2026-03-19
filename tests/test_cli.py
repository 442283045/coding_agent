"""Tests for the CLI entrypoint."""

from rich.console import Console
from typer.testing import CliRunner

import coding_agent.cli as cli_module

runner = CliRunner()


def test_root_command_starts_interactive_session(monkeypatch, tmp_path) -> None:
    """Running the root command without arguments should start interactive mode."""

    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, working_dir: str, model: str, debug: bool) -> None:
            captured["working_dir"] = working_dir
            captured["model"] = model
            captured["debug"] = debug

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
    assert captured["interactive_started"] is True
    assert "Coding Agent" in result.output
    assert f"Workspace: {tmp_path.resolve()}" in result.output


def test_root_command_accepts_workspace_option(monkeypatch, tmp_path) -> None:
    """The root interactive command should accept --workspace/-w."""

    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, working_dir: str, model: str, debug: bool) -> None:
            captured["working_dir"] = working_dir
            captured["model"] = model
            captured["debug"] = debug

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
    assert captured["interactive_started"] is True
    assert f"Workspace: {tmp_path}" in result.output


def test_chat_banner_uses_ascii_title(monkeypatch, tmp_path) -> None:
    """The chat banner should render safely on non-UTF-8 Windows consoles."""

    class FakeAgent:
        def __init__(self, working_dir: str, model: str, debug: bool) -> None:
            self.working_dir = working_dir
            self.model = model
            self.debug = debug

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


def test_root_workspace_banner_preserves_relative_workspace_option(monkeypatch, tmp_path) -> None:
    """The banner should show the user-provided -w value while resolving the real directory."""

    captured: dict[str, object] = {}
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class FakeAgent:
        def __init__(self, working_dir: str, model: str, debug: bool) -> None:
            captured["working_dir"] = working_dir

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
    assert "Workspace: workspace" in result.output


def test_run_accepts_workspace_option(monkeypatch, tmp_path) -> None:
    """The run command should accept --workspace/-w as a path alias."""

    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, working_dir: str, model: str, debug: bool) -> None:
            captured["working_dir"] = working_dir
            captured["model"] = model
            captured["debug"] = debug

        def run_once(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "done"

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)

    result = runner.invoke(cli_module.app, ["run", "hello", "-w", str(tmp_path)])

    assert result.exit_code == 0
    assert captured["working_dir"] == str(tmp_path.resolve())
    assert captured["prompt"] == "hello"
    assert "done" in result.output


def test_run_renders_markdown_output(monkeypatch, tmp_path) -> None:
    """The run command should render final output as Markdown."""

    rendered_markdown: list[str] = []

    class FakeMarkdown:
        def __init__(self, content: str) -> None:
            self.content = content
            rendered_markdown.append(content)

    class FakeAgent:
        def __init__(self, working_dir: str, model: str, debug: bool) -> None:
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
    tmp_path,
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
