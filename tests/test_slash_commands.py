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
from coding_agent.skills import SkillCatalog, SkillManager


def _empty_skill_catalog() -> SkillCatalog:
    """Return an empty skill catalog for slash-command tests."""
    return SkillCatalog()


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
            list_skills=_empty_skill_catalog,
            reload_skills=_empty_skill_catalog,
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
            list_skills=_empty_skill_catalog,
            reload_skills=_empty_skill_catalog,
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
            list_skills=_empty_skill_catalog,
            reload_skills=_empty_skill_catalog,
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

    assert [completion.display_text for completion in completions] == ["/mcp", "/skills"]
    assert (
        completions[0].display_meta_text
        == "Inspect, add, remove, and reload MCP server configuration."
    )
    assert completions[1].display_meta_text == "Inspect and rescan installed workspace skills."


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


def test_skills_slash_command_lists_current_skills(tmp_path: Path) -> None:
    """`/skills` should render the discovered workspace skills."""

    skill_root = tmp_path / ".codex" / "skills" / "reviewer"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "# Reviewer\n\nReview code changes for regressions.\n",
        encoding="utf-8",
    )

    skill_manager = SkillManager(working_dir=tmp_path)
    slash_commands = create_default_slash_command_registry()
    app_settings = Settings(_env_file=None)

    result = slash_commands.execute(
        "/skills",
        ctx=SlashCommandContext(
            working_dir=tmp_path,
            settings=app_settings,
            reload_mcp_tools=lambda: [],
            list_skills=skill_manager.discover,
            reload_skills=skill_manager.discover,
        ),
    )

    assert result is not None
    assert "# Skills" in result.output
    assert "Discovered skills: 1" in result.output
    assert "`Reviewer`" in result.output


def test_slash_completer_suggests_skills_subcommands() -> None:
    """Typing `/skills ` should show candidate `/skills` actions."""

    completions = (
        create_default_slash_command_registry()
        .build_completer()
        .get_completions(
            Document(text="/skills ", cursor_position=8),
            CompleteEvent(completion_requested=False),
        )
    )

    assert [completion.text for completion in completions] == ["list", "help", "reload"]
