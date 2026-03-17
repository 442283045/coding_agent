"""Tests for the CLI entrypoint."""

from typer.testing import CliRunner

import coding_agent.cli as cli_module

runner = CliRunner()


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

    result = runner.invoke(cli_module.app, ["chat", str(tmp_path)])

    assert result.exit_code == 0
    assert "Coding Agent" in result.output
    assert "🤖 Coding Agent" not in result.output
