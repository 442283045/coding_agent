"""Core agent implementation with ReAct loop."""

import json
from collections.abc import Awaitable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from jinja2 import Template
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from coding_agent.agent.slash_commands import (
    SlashCommandContext,
    SlashCommandResult,
    create_default_slash_command_registry,
)
from coding_agent.config import settings
from coding_agent.llm.client import LLMClient
from coding_agent.mcp import MCPManager
from coding_agent.shell_environment import ShellProfile, detect_shell_profile
from coding_agent.skills import SkillCatalog, SkillManager
from coding_agent.tools.registry import ToolRegistry, registry

console = Console()
PROMPT_STYLE = Style.from_dict({"prompt": "bold green"})
T = TypeVar("T")


class Agent:
    """ReAct-based coding agent."""

    def __init__(
        self,
        working_dir: str,
        model: str | None = None,
        debug: bool = False,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.model = model or settings.default_model
        self.debug = debug
        self.history: list[dict[str, Any]] = []
        self._interaction_log_path: Path | None = None
        self._async_runner: object | None = None
        self.registry: ToolRegistry = ToolRegistry()
        self.mcp_manager = MCPManager(settings.load_mcp_servers())
        self.shell_profile = detect_shell_profile()
        self.skill_manager = SkillManager(working_dir=self.working_dir)
        self.skill_catalog: SkillCatalog = self.skill_manager.discover()
        self.slash_commands = create_default_slash_command_registry()

        # Initialize tool context
        self.tool_context = {
            "working_dir": str(self.working_dir),
            "debug": debug,
        }

        # Initialize tools
        self._init_tools()
        self.llm = LLMClient(model=self.model, debug=debug, tool_registry=self.registry)

        # Setup prompt session with history
        settings.ensure_config_dir()
        history_file = settings.config_dir / "cli_history"
        self.session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.slash_commands.build_completer(),
            complete_while_typing=True,
        )

    def _build_base_registry(self) -> ToolRegistry:
        """Build a fresh registry containing only the built-in local tools."""
        # Import tools to trigger registration
        from coding_agent.tools import code_tools, file_tools, shell_tools

        _ = (code_tools, file_tools, shell_tools)
        return registry.copy()

    def _start_mcp_manager(
        self,
        mcp_manager: MCPManager,
        tool_registry: ToolRegistry,
    ) -> list[str]:
        """Start an MCP manager against a specific tool registry."""
        if not mcp_manager.enabled:
            return []

        return self._run_async(mcp_manager.start(tool_registry))

    def _get_async_runner(self) -> Any:
        """Get or create the shared asyncio runner for this agent."""
        runner = getattr(self, "_async_runner", None)
        if runner is not None:
            return runner

        import asyncio

        runner = asyncio.Runner()
        self._async_runner = runner
        return runner

    def _run_async(self, awaitable: Awaitable[T]) -> T:
        """Run an awaitable on the agent's shared event loop."""
        runner = self._get_async_runner()
        return cast(T, runner.run(awaitable))

    def _sync_llm_tool_registry(self) -> None:
        """Keep the LLM client pointed at the active tool registry."""
        llm = getattr(self, "llm", None)
        if llm is not None:
            llm.tool_registry = self.registry

    def _log_loaded_tools(self) -> None:
        """Emit a debug summary of the active tool set."""
        if self.debug:
            tools = self.registry.list_tools()
            console.print(f"[dim]Loaded {len(tools)} tools[/dim]")

    def _display_mcp_startup_success(self, tool_names: Sequence[str]) -> None:
        """Show a user-visible MCP startup summary."""
        if not self.mcp_manager.enabled:
            return

        server_names = ", ".join(f"`{name}`" for name in self.mcp_manager.server_names)
        console.print(f"[green]MCP configured successfully.[/green] Loaded servers: {server_names}")

        if not tool_names:
            console.print("[yellow]MCP is connected, but no tools were exposed.[/yellow]")
            return

        console.print("[green]Available MCP tools:[/green]")
        for tool_name in sorted(tool_names):
            console.print(f"[dim]- {tool_name}[/dim]")

    def _display_mcp_startup_failure(self, error: Exception) -> None:
        """Show a user-visible MCP startup failure summary."""
        if not self.mcp_manager.enabled:
            return

        server_names = ", ".join(f"`{name}`" for name in self.mcp_manager.server_names)
        console.print(f"[red]MCP configuration failed.[/red] Configured servers: {server_names}")
        console.print(f"[red]Reason:[/red] {error}")

    def _init_tools(self) -> None:
        """Initialize and register all tools."""
        self.registry = self._build_base_registry()
        registered_mcp_tools: list[str] = []
        if self.mcp_manager.enabled:
            try:
                registered_mcp_tools = self._start_mcp_manager(self.mcp_manager, self.registry)
            except Exception as error:
                self._display_mcp_startup_failure(error)
                raise
            else:
                self._display_mcp_startup_success(registered_mcp_tools)
        self._sync_llm_tool_registry()
        self._log_loaded_tools()

    def reload_mcp_tools(self) -> list[str]:
        """Reload MCP servers without dropping the current session on failure."""
        next_registry = self._build_base_registry()
        next_manager = MCPManager(settings.load_mcp_servers())

        try:
            registered_names = self._start_mcp_manager(next_manager, next_registry)
        except Exception:
            if next_manager.enabled:
                self._run_async(next_manager.close())
            raise

        previous_manager = self.mcp_manager
        self.registry = next_registry
        self.mcp_manager = next_manager
        self._sync_llm_tool_registry()

        if previous_manager.enabled:
            self._run_async(previous_manager.close())

        self._log_loaded_tools()
        return registered_names

    def _load_system_prompt(self) -> str:
        """Load and render system prompt template."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "system.j2"
        skill_catalog = self.list_skills()
        skill_entries = skill_catalog.prompt_entries(self.working_dir)

        if prompt_path.exists():
            template = Template(prompt_path.read_text(encoding="utf-8"))
            return template.render(
                working_dir=str(self.working_dir),
                tools=self.registry.list_tools(),
                shell_profile=self.get_shell_profile(),
                skill_search_roots=skill_catalog.root_display_paths(self.working_dir),
                skills=skill_entries,
            )

        # Fallback system prompt
        return self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if template is not found."""
        return (
            "You are a helpful AI coding assistant. You are working in directory: "
            f"{self.working_dir}\n\n"
            "You have access to various tools to help you complete tasks. Use them when "
            "needed.\n\n"
            "Available tools:\n"
        )

    def _build_messages(self) -> list[dict[str, Any]]:
        """Build message list with system prompt and history."""
        system_prompt = self._load_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        return messages

    def _summarize_tool_arguments(self, arguments: dict[str, Any]) -> str:
        """Create a compact user-facing summary of tool arguments."""
        try:
            summary = json.dumps(arguments, ensure_ascii=False, default=str)
        except TypeError:
            summary = str(arguments)

        max_length = 160
        if len(summary) > max_length:
            return summary[: max_length - 3] + "..."
        return summary

    def _ensure_interaction_log_file(self) -> Path:
        """Create the per-session LLM interaction log file on first user submission."""
        if self._interaction_log_path is not None:
            return self._interaction_log_path

        settings.ensure_config_dir()
        log_dir = settings.config_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        log_path = (log_dir / f"interactive-{timestamp}.log").resolve()
        log_path.touch(exist_ok=True)

        self._interaction_log_path = log_path
        self.llm.set_log_path(log_path)

        console.print(f"[dim]LLM log file: {log_path}[/dim]")
        return log_path

    def _build_slash_command_context(self) -> SlashCommandContext:
        """Build the runtime context passed to slash commands."""
        return SlashCommandContext(
            working_dir=self.working_dir,
            settings=settings,
            reload_mcp_tools=self.reload_mcp_tools,
            list_skills=self.list_skills,
            reload_skills=self.reload_skills,
        )

    def list_skills(self) -> SkillCatalog:
        """Return the currently discovered workspace skill catalog."""
        skill_catalog = getattr(self, "skill_catalog", None)
        if isinstance(skill_catalog, SkillCatalog):
            return skill_catalog
        return SkillCatalog()

    def get_shell_profile(self) -> ShellProfile:
        """Return the shell profile used for command generation and execution."""
        shell_profile = getattr(self, "shell_profile", None)
        if isinstance(shell_profile, ShellProfile):
            return shell_profile
        return detect_shell_profile()

    def reload_skills(self) -> SkillCatalog:
        """Rescan workspace-local skills and update the prompt context."""
        skill_manager = getattr(self, "skill_manager", None)
        if skill_manager is None:
            self.skill_catalog = SkillCatalog()
            return self.skill_catalog

        self.skill_catalog = skill_manager.discover()
        return self.skill_catalog

    def _dispatch_slash_command(self, user_input: str) -> SlashCommandResult | None:
        """Dispatch a slash command if the input is handled locally."""
        slash_commands = getattr(self, "slash_commands", None)
        if slash_commands is None:
            return None

        return cast(
            SlashCommandResult | None,
            slash_commands.execute(
                user_input,
                ctx=self._build_slash_command_context(),
            ),
        )

    def _display_slash_command_result(self, content: str) -> None:
        """Render local slash command output in the terminal."""
        console.print("\n[bold cyan]Command:[/bold cyan]")
        console.print(Markdown(content))

    def run_interactive(self) -> None:
        """Run interactive chat session."""
        while True:
            try:
                # Get user input with prompt_toolkit
                user_input = self.session.prompt(
                    [("class:prompt", "You: ")],
                    style=PROMPT_STYLE,
                )

                if not user_input.strip():
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                slash_result = self._dispatch_slash_command(user_input)
                if slash_result is not None:
                    self._display_slash_command_result(slash_result.output)
                    continue

                # Process the message
                self._process_message(user_input)

            except KeyboardInterrupt:
                continue
            except EOFError:
                break

    def _process_message(self, user_input: str) -> None:
        """Process a single user message through the ReAct loop."""
        self._ensure_interaction_log_file()
        self.history.append({"role": "user", "content": user_input})

        max_iterations = settings.max_iterations

        for iteration in range(max_iterations):
            stream_state: dict[str, object] = {
                "content": "",
                "started": False,
                "live": None,
            }

            def on_content_chunk(
                chunk: str,
                state: dict[str, object] = stream_state,
            ) -> None:
                state["content"] = f"{state['content']}{chunk}"
                renderable = Markdown(str(state["content"]))

                if not bool(state["started"]):
                    console.print("\n[bold blue]Agent:[/bold blue]")
                    live = Live(renderable, console=console, refresh_per_second=8, transient=False)
                    live.start()
                    state["live"] = live
                    state["started"] = True
                    return

                current_live = cast(Live | None, state["live"])
                if isinstance(current_live, Live):
                    current_live.update(renderable, refresh=True)

            try:
                response = self._run_async(
                    self.llm.chat_with_tools(
                        self._build_messages(),
                        on_content_chunk=on_content_chunk,
                    )
                )
            finally:
                current_live = cast(Live | None, stream_state["live"])
                if isinstance(current_live, Live):
                    current_live.stop()
                    console.print()

            content = str(response.get("content", ""))
            tool_calls = response.get("tool_calls", [])
            reasoning_content = response.get("reasoning_content")

            if self.debug:
                console.print(f"[dim]Iteration {iteration + 1}: {len(tool_calls)} tool calls[/dim]")

            # Handle tool calls
            if tool_calls:
                # Add assistant message with tool calls
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                }
                if isinstance(reasoning_content, str) and reasoning_content:
                    assistant_message["reasoning_content"] = reasoning_content

                # Format tool calls for the message
                if "claude" in self.model.lower():
                    # Anthropic format
                    assistant_message["content"] = [
                        {"type": "text", "text": content or ""},
                    ]
                    for tc in tool_calls:
                        assistant_message["content"].append(
                            {
                                "type": "tool_use",
                                "id": tc["id"],
                                "name": tc["name"],
                                "input": json.loads(tc["arguments"]),
                            }
                        )
                else:
                    # OpenAI format
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ]

                self.history.append(assistant_message)

                # Execute tools
                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call)

                    # Add tool response
                    if "claude" in self.model.lower():
                        self.history.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_call["id"],
                                        "content": result,
                                    }
                                ],
                            }
                        )
                    else:
                        self.history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": result,
                            }
                        )
            else:
                # No tool calls - display the response
                if not stream_state["started"]:
                    self._display_response(content)
                self.history.append(
                    {
                        "role": "assistant",
                        "content": content,
                        **(
                            {"reasoning_content": reasoning_content}
                            if isinstance(reasoning_content, str) and reasoning_content
                            else {}
                        ),
                    }
                )
                break
        else:
            console.print("[yellow]Warning: Reached maximum iterations.[/yellow]")

    def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a tool call."""
        tool_name = tool_call["name"]
        tool = self.registry.get(tool_name)

        if not tool:
            return f"Error: Tool '{tool_name}' not found."

        try:
            args = json.loads(tool_call["arguments"])
            console.print(
                f"[dim]Calling tool {tool_name}: {self._summarize_tool_arguments(args)}[/dim]"
            )
            # Add context to arguments
            args["ctx"] = self.tool_context

            result = self._run_async(tool.execute(**args))
            status = "failed" if result.startswith("Error") else "completed"
            console.print(f"[dim]Tool {tool_name} {status}[/dim]")

            # Truncate result if too long
            max_result_len = 10000
            if len(result) > max_result_len:
                result = result[:max_result_len] + "\n... [truncated]"

            return result

        except json.JSONDecodeError as e:
            return f"Error: Invalid tool arguments - {e}"
        except Exception as e:
            if self.debug:
                raise
            return f"Error executing tool: {e}"

    def _display_response(self, content: str) -> None:
        """Display assistant response with formatting."""
        console.print("\n[bold blue]Agent:[/bold blue]")
        console.print(Markdown(content))

    def close(self) -> None:
        """Close session-scoped resources like MCP clients."""
        try:
            if self.mcp_manager.enabled:
                self._run_async(self.mcp_manager.close())
        finally:
            runner = getattr(self, "_async_runner", None)
            if runner is not None:
                runner.close()
                self._async_runner = None

    def run_once(self, prompt: str) -> str:
        """Execute a single prompt and return the result."""
        try:
            slash_result = self._dispatch_slash_command(prompt)
            if slash_result is not None:
                return slash_result.output

            self.history.append({"role": "user", "content": prompt})

            response = self._run_async(self.llm.chat_with_tools(self._build_messages()))

            content = str(response.get("content", ""))
            tool_calls = response.get("tool_calls", [])
            reasoning_content = response.get("reasoning_content")

            max_iterations = settings.max_iterations
            iteration = 0

            while tool_calls and iteration < max_iterations:
                iteration += 1

                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ],
                }
                if isinstance(reasoning_content, str) and reasoning_content:
                    assistant_message["reasoning_content"] = reasoning_content
                self.history.append(assistant_message)

                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call)
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result,
                        }
                    )

                response = self._run_async(self.llm.chat_with_tools(self._build_messages()))
                content = str(response.get("content", ""))
                tool_calls = response.get("tool_calls", [])
                reasoning_content = response.get("reasoning_content")

            final_message: dict[str, Any] = {"role": "assistant", "content": content}
            if isinstance(reasoning_content, str) and reasoning_content:
                final_message["reasoning_content"] = reasoning_content
            self.history.append(final_message)
            return content
        finally:
            self.close()
