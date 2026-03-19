"""Core agent implementation with ReAct loop."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from jinja2 import Template
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from coding_agent.config import settings
from coding_agent.llm.client import LLMClient
from coding_agent.tools.registry import registry

console = Console()
PROMPT_STYLE = Style.from_dict({"prompt": "bold green"})


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
        self.llm = LLMClient(model=self.model, debug=debug)
        self.history: list[dict[str, Any]] = []
        self._interaction_log_path: Path | None = None

        # Initialize tool context
        self.tool_context = {
            "working_dir": str(self.working_dir),
            "debug": debug,
        }

        # Initialize tools
        self._init_tools()

        # Setup prompt session with history
        settings.ensure_config_dir()
        history_file = settings.config_dir / "cli_history"
        self.session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
        )

    def _init_tools(self) -> None:
        """Initialize and register all tools."""
        # Import tools to trigger registration
        from coding_agent.tools import code_tools, file_tools, shell_tools

        _ = (code_tools, file_tools, shell_tools)

        if self.debug:
            tools = registry.list_tools()
            console.print(f"[dim]Loaded {len(tools)} tools[/dim]")

    def _load_system_prompt(self) -> str:
        """Load and render system prompt template."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "system.j2"

        if prompt_path.exists():
            template = Template(prompt_path.read_text())
            return template.render(
                working_dir=str(self.working_dir),
                tools=registry.list_tools(),
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

            import asyncio

            try:
                response = asyncio.run(
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
        tool = registry.get(tool_name)

        if not tool:
            return f"Error: Tool '{tool_name}' not found."

        try:
            args = json.loads(tool_call["arguments"])
            console.print(
                f"[dim]Calling tool {tool_name}: {self._summarize_tool_arguments(args)}[/dim]"
            )
            # Add context to arguments
            args["ctx"] = self.tool_context

            import asyncio

            result = asyncio.run(tool.execute(**args))
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

    def run_once(self, prompt: str) -> str:
        """Execute a single prompt and return the result."""
        self.history.append({"role": "user", "content": prompt})

        import asyncio

        response = asyncio.run(self.llm.chat_with_tools(self._build_messages()))

        content = str(response.get("content", ""))
        tool_calls = response.get("tool_calls", [])
        reasoning_content = response.get("reasoning_content")

        max_iterations = settings.max_iterations
        iteration = 0

        while tool_calls and iteration < max_iterations:
            iteration += 1

            # Add assistant message
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

            # Execute tools
            for tool_call in tool_calls:
                result = self._execute_tool(tool_call)
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )

            # Get next response
            response = asyncio.run(self.llm.chat_with_tools(self._build_messages()))
            content = str(response.get("content", ""))
            tool_calls = response.get("tool_calls", [])
            reasoning_content = response.get("reasoning_content")

        # Final response
        final_message: dict[str, Any] = {"role": "assistant", "content": content}
        if isinstance(reasoning_content, str) and reasoning_content:
            final_message["reasoning_content"] = reasoning_content
        self.history.append(final_message)
        return content
