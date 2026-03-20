"""Tests for local interactive slash commands."""

import json
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from coding_agent.agent.slash_commands import (
    SlashCommandContext,
    create_default_slash_command_registry,
)
from coding_agent.config import Settings


def test_mcp_slash_command_adds_server_to_disk_and_reloads(tmp_path: Path) -> None:
    """`/mcp add` should persist a server and refresh the current session."""

    reload_calls: list[bool] = []

    def reload_mcp_tools() -> list[str]:
        reload_calls.append(True)
        return ["mcp__filesystem__read_file"]

    slash_commands = create_default_slash_command_registry()
    app_settings = Settings(_env_file=None, mcp_config_path=tmp_path / "mcp.json")
    config_json = json.dumps({"command": "npx", "args": ["-y", "server", str(tmp_path)]})
    result = slash_commands.execute(
        f"/mcp add filesystem {config_json}",
        ctx=SlashCommandContext(
            working_dir=tmp_path,
            settings=app_settings,
            reload_mcp_tools=reload_mcp_tools,
        ),
    )

    assert result is not None
    assert "Added MCP server `filesystem`." in result.output
    assert "Saved MCP configuration to" in result.output
    assert "Registered 1 tool(s)" in result.output
    assert reload_calls == [True]

    servers = app_settings.load_mcp_servers()
    assert set(servers) == {"filesystem"}
    assert servers["filesystem"].command == "npx"
    assert servers["filesystem"].args[-1] == str(tmp_path)


def test_mcp_slash_command_updates_in_memory_json_override(tmp_path: Path) -> None:
    """`/mcp add` should update the in-memory env override when that source is active."""

    slash_commands = create_default_slash_command_registry()
    app_settings = Settings(_env_file=None, mcp_servers_json='{"mcpServers":{}}')
    result = slash_commands.execute(
        '/mcp add docs {"url":"https://example.com/mcp","transport":"http"}',
        ctx=SlashCommandContext(
            working_dir=tmp_path,
            settings=app_settings,
            reload_mcp_tools=lambda: [],
        ),
    )

    assert result is not None
    assert "Updated `AGENT_MCP_SERVERS_JSON` in memory for this session." in result.output
    assert "not written to disk" in result.output

    servers = app_settings.load_mcp_servers()
    assert set(servers) == {"docs"}
    assert servers["docs"].url == "https://example.com/mcp"
    assert servers["docs"].transport == "http"


def test_mcp_slash_command_lists_current_servers(tmp_path: Path) -> None:
    """`/mcp` should show the current effective configuration and usage."""

    app_settings = Settings(
        _env_file=None,
        mcp_servers_json='{"mcpServers":{"docs":{"url":"https://example.com/mcp"}}}',
    )
    slash_commands = create_default_slash_command_registry()

    result = slash_commands.execute(
        "/mcp",
        ctx=SlashCommandContext(
            working_dir=tmp_path,
            settings=app_settings,
            reload_mcp_tools=lambda: [],
        ),
    )

    assert result is not None
    assert "# MCP Configuration" in result.output
    assert "Configured servers: 1" in result.output
    assert "`docs`: `streamable-http` -> `https://example.com/mcp`" in result.output
    assert "Usage:" in result.output


def test_slash_completer_suggests_top_level_commands() -> None:
    """Typing `/` should show available slash commands."""

    completions = (
        create_default_slash_command_registry()
        .build_completer()
        .get_completions(
            Document(text="/", cursor_position=1),
            CompleteEvent(completion_requested=False),
        )
    )

    assert [completion.display_text for completion in completions] == ["/mcp"]
    assert (
        completions[0].display_meta_text
        == "Inspect, add, remove, and reload MCP server configuration."
    )


def test_slash_completer_suggests_mcp_subcommands() -> None:
    """Typing `/mcp ` should show candidate `/mcp` actions."""

    completions = (
        create_default_slash_command_registry()
        .build_completer()
        .get_completions(
            Document(text="/mcp ", cursor_position=5),
            CompleteEvent(completion_requested=False),
        )
    )

    assert [completion.text for completion in completions] == [
        "list",
        "help",
        "add",
        "set",
        "remove",
        "rm",
        "reload",
    ]
