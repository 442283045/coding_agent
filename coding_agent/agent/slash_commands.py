"""Interactive slash commands handled locally by the CLI agent."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from pydantic import ValidationError

from coding_agent.config import MCPServerConfig, Settings


@dataclass(slots=True)
class SlashCommandContext:
    """Runtime context passed to slash command handlers."""

    working_dir: Path
    settings: Settings
    reload_mcp_tools: Callable[[], list[str]]


@dataclass(slots=True)
class SlashCommandResult:
    """Structured result returned by a slash command."""

    output: str


@dataclass(slots=True)
class SlashCommand:
    """Registered slash command metadata."""

    name: str
    description: str
    handler: Callable[[SlashCommandContext, str], SlashCommandResult]
    subcommands: tuple[SlashCommandOption, ...] = ()


@dataclass(frozen=True, slots=True)
class SlashCommandOption:
    """Metadata for slash command completion options."""

    value: str
    description: str


class SlashCommandError(ValueError):
    """Raised when a slash command cannot be executed."""


class SlashCommandRegistry:
    """Registry and dispatcher for interactive slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        """Register a slash command handler."""
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        """Look up a registered slash command."""
        return self._commands.get(name)

    def list_commands(self) -> list[SlashCommand]:
        """List registered slash commands."""
        return sorted(self._commands.values(), key=lambda item: item.name)

    def build_completer(self) -> SlashCommandCompleter:
        """Build a prompt_toolkit completer for slash commands."""
        return SlashCommandCompleter(self)

    def execute(
        self,
        user_input: str,
        *,
        ctx: SlashCommandContext,
    ) -> SlashCommandResult | None:
        """Execute a slash command when the input starts with '/'."""
        stripped = user_input.strip()
        if not stripped.startswith("/"):
            return None

        command_text = stripped[1:]
        if not command_text:
            return SlashCommandResult(self.format_help())

        command_name, _, arguments = command_text.partition(" ")
        command = self._commands.get(command_name.lower())
        if command is None:
            return SlashCommandResult(
                f"Unknown slash command `/{command_name}`.\n\n{self.format_help()}"
            )

        try:
            return command.handler(ctx, arguments.strip())
        except SlashCommandError as exc:
            return SlashCommandResult(str(exc))

    def format_help(self) -> str:
        """Render the list of available slash commands."""
        if not self._commands:
            return "No slash commands are registered."

        lines = ["Available slash commands:"]
        for command in self.list_commands():
            lines.append(f"- `/{command.name}`: {command.description}")
        return "\n".join(lines)


class SlashCommandCompleter(Completer):
    """Prompt completer that suggests slash commands and subcommands."""

    def __init__(self, registry: SlashCommandRegistry) -> None:
        self._registry = registry

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> list[Completion]:
        """Return completions for slash-command input."""
        _ = complete_event
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return []

        command_text = text[1:]
        if " " not in command_text:
            return self._complete_commands(command_text)

        command_name, _, remainder = command_text.partition(" ")
        command = self._registry.get(command_name.lower())
        if command is None or not command.subcommands:
            return []

        remainder_prefix = remainder.lstrip()
        if not remainder_prefix:
            return self._complete_subcommands("", command.subcommands)

        if remainder.endswith(" "):
            return []

        if " " in remainder_prefix:
            return []

        return self._complete_subcommands(remainder_prefix, command.subcommands)

    def _complete_commands(self, partial: str) -> list[Completion]:
        """Complete top-level slash commands."""
        completions: list[Completion] = []
        for command in self._registry.list_commands():
            if not command.name.startswith(partial.lower()):
                continue
            completions.append(
                Completion(
                    text=command.name,
                    start_position=-len(partial),
                    display=f"/{command.name}",
                    display_meta=command.description,
                )
            )
        return completions

    def _complete_subcommands(
        self,
        partial: str,
        options: tuple[SlashCommandOption, ...],
    ) -> list[Completion]:
        """Complete the first subcommand token."""
        completions: list[Completion] = []
        for option in options:
            if not option.value.startswith(partial.lower()):
                continue
            completions.append(
                Completion(
                    text=option.value,
                    start_position=-len(partial),
                    display=option.value,
                    display_meta=option.description,
                )
            )
        return completions


def create_default_slash_command_registry() -> SlashCommandRegistry:
    """Create the default slash command registry for interactive sessions."""
    registry = SlashCommandRegistry()
    registry.register(
        SlashCommand(
            name="mcp",
            description="Inspect, add, remove, and reload MCP server configuration.",
            handler=_handle_mcp_command,
            subcommands=(
                SlashCommandOption("list", "Show the current MCP configuration."),
                SlashCommandOption("help", "Show `/mcp` usage."),
                SlashCommandOption("add", "Add a new MCP server from JSON."),
                SlashCommandOption("set", "Alias for `add`."),
                SlashCommandOption("remove", "Remove a configured MCP server."),
                SlashCommandOption("rm", "Alias for `remove`."),
                SlashCommandOption("reload", "Reload MCP tools for this session."),
            ),
        )
    )
    return registry


def _handle_mcp_command(ctx: SlashCommandContext, arguments: str) -> SlashCommandResult:
    """Handle the /mcp slash command."""
    if not arguments:
        return SlashCommandResult(_render_mcp_overview(ctx))

    action, _, remainder = arguments.partition(" ")
    remainder = remainder.strip()

    match action.lower():
        case "list":
            return SlashCommandResult(_render_mcp_overview(ctx))
        case "help":
            return SlashCommandResult(_render_mcp_help(ctx))
        case "add" | "set":
            return SlashCommandResult(_add_mcp_server(ctx, remainder))
        case "remove" | "rm":
            return SlashCommandResult(_remove_mcp_server(ctx, remainder))
        case "reload":
            return SlashCommandResult(_reload_mcp_servers(ctx))
        case _:
            raise SlashCommandError(f"Unknown `/mcp` action `{action}`.\n\n{_render_mcp_help(ctx)}")


def _render_mcp_overview(ctx: SlashCommandContext) -> str:
    """Render the current effective MCP configuration and command help."""
    servers = ctx.settings.load_mcp_servers()
    lines = [
        "# MCP Configuration",
        "",
        f"Config source: {_describe_mcp_config_source(ctx.settings)}",
        f"Configured servers: {len(servers)}",
    ]

    if servers:
        lines.extend(["", "Servers:"])
        for server_name, server_config in sorted(servers.items()):
            lines.append(f"- `{server_name}`: {_describe_server(server_config)}")
    else:
        lines.extend(["", "No MCP servers are configured yet."])

    lines.extend(["", _render_mcp_help(ctx)])
    return "\n".join(lines)


def _render_mcp_help(ctx: SlashCommandContext) -> str:
    """Render /mcp usage instructions."""
    example_config = {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(ctx.working_dir)],
    }
    example_json = json.dumps(example_config, ensure_ascii=False)
    lines = [
        "Usage:",
        "- `/mcp`",
        "- `/mcp add <name> <json>`",
        "- `/mcp remove <name>`",
        "- `/mcp reload`",
        "",
        "Example:",
        f"- `/mcp add filesystem {example_json}`",
    ]
    if ctx.settings.mcp_servers_json is not None:
        lines.extend(
            [
                "",
                (
                    "Note: `AGENT_MCP_SERVERS_JSON` is active, so slash edits only affect "
                    "this session."
                ),
            ]
        )
    return "\n".join(lines)


def _add_mcp_server(ctx: SlashCommandContext, remainder: str) -> str:
    """Add or replace an MCP server configuration."""
    server_name, separator, raw_config = remainder.partition(" ")
    if not server_name or not separator or not raw_config.strip():
        raise SlashCommandError(
            "Usage: `/mcp add <name> <json>`\n\n"
            "Example: `/mcp add filesystem "
            '{"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","."]}`'
        )

    try:
        server_config = MCPServerConfig.model_validate(json.loads(raw_config))
    except json.JSONDecodeError as exc:
        raise SlashCommandError(f"Invalid JSON for MCP server `{server_name}`: {exc.msg}") from exc
    except ValidationError as exc:
        raise SlashCommandError(f"Invalid MCP server config for `{server_name}`.\n\n{exc}") from exc

    servers = ctx.settings.load_mcp_servers()
    action = "Updated" if server_name in servers else "Added"
    servers[server_name] = server_config

    persistence_message = _persist_mcp_servers(ctx.settings, servers)
    reload_message = _reload_current_session(ctx)
    return f"{action} MCP server `{server_name}`.\n\n{persistence_message}\n\n{reload_message}"


def _remove_mcp_server(ctx: SlashCommandContext, remainder: str) -> str:
    """Remove an MCP server configuration."""
    server_name = remainder.strip()
    if not server_name:
        raise SlashCommandError("Usage: `/mcp remove <name>`")

    servers = ctx.settings.load_mcp_servers()
    if server_name not in servers:
        raise SlashCommandError(f"MCP server `{server_name}` is not configured.")

    del servers[server_name]

    persistence_message = _persist_mcp_servers(ctx.settings, servers)
    reload_message = _reload_current_session(ctx)
    return f"Removed MCP server `{server_name}`.\n\n{persistence_message}\n\n{reload_message}"


def _reload_mcp_servers(ctx: SlashCommandContext) -> str:
    """Reload MCP tools for the current session."""
    return _reload_current_session(ctx)


def _reload_current_session(ctx: SlashCommandContext) -> str:
    """Reload the effective MCP configuration into the running agent."""
    try:
        registered_tools = ctx.reload_mcp_tools()
    except Exception as exc:  # pragma: no cover - surfaced as user-facing text
        raise SlashCommandError(f"Failed to reload MCP tools: {exc}") from exc

    tool_count = len(registered_tools)
    if tool_count == 0:
        return "Reloaded MCP tools for the current session. No MCP tools are currently registered."

    tool_list = ", ".join(f"`{name}`" for name in registered_tools)
    return (
        f"Reloaded MCP tools for the current session. Registered {tool_count} tool(s): {tool_list}"
    )


def _persist_mcp_servers(settings_obj: Settings, servers: dict[str, MCPServerConfig]) -> str:
    """Persist MCP servers to the effective configuration source."""
    ordered_servers = dict(sorted(servers.items()))
    if settings_obj.mcp_servers_json is not None:
        settings_obj.update_mcp_servers_json(ordered_servers)
        return (
            "Updated `AGENT_MCP_SERVERS_JSON` in memory for this session. "
            "The change was not written to disk."
        )

    config_path = settings_obj.save_mcp_servers(ordered_servers)
    return f"Saved MCP configuration to `{config_path}`."


def _describe_mcp_config_source(settings_obj: Settings) -> str:
    """Describe where MCP configuration is loaded from."""
    if settings_obj.mcp_servers_json is not None:
        return "`AGENT_MCP_SERVERS_JSON`"
    return f"`{settings_obj.effective_mcp_config_path}`"


def _describe_server(server_config: MCPServerConfig) -> str:
    """Render a compact one-line summary of an MCP server."""
    if server_config.command:
        command_line = " ".join([server_config.command, *server_config.args])
        return f"`stdio` -> `{command_line}`"

    transport = server_config.transport or "streamable-http"
    return f"`{transport}` -> `{server_config.url}`"


default_slash_commands = create_default_slash_command_registry()

__all__ = [
    "SlashCommand",
    "SlashCommandCompleter",
    "SlashCommandContext",
    "SlashCommandOption",
    "SlashCommandRegistry",
    "SlashCommandResult",
    "create_default_slash_command_registry",
    "default_slash_commands",
]
