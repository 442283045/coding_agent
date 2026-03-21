"""Tests for the core agent loop."""

import asyncio
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from coding_agent.agent.core import PROMPT_STYLE, Agent
from coding_agent.agent.slash_commands import SlashCommandCompleter, SlashCommandResult
from coding_agent.config import Settings
from coding_agent.shell_environment import ShellProfile
from coding_agent.skills import SkillCatalog
from coding_agent.tools.registry import ToolRegistry


def test_run_interactive_uses_prompt_toolkit_style(monkeypatch) -> None:
    """The interactive prompt should pass a prompt_toolkit Style instance."""

    prompt_calls: list[dict[str, object]] = []

    class FakeSession:
        def prompt(self, *args: object, **kwargs: object) -> str:
            prompt_calls.append({"args": args, "kwargs": kwargs})
            return "quit"

    agent = Agent.__new__(Agent)
    agent.session = FakeSession()

    monkeypatch.setattr("coding_agent.agent.core.console.print", lambda *args, **kwargs: None)

    Agent.run_interactive(agent)

    assert len(prompt_calls) == 1
    prompt_kwargs = prompt_calls[0]["kwargs"]
    assert prompt_kwargs["style"] is PROMPT_STYLE
    assert hasattr(prompt_kwargs["style"], "invalidation_hash")


def test_agent_passes_debug_flag_to_llm_client(monkeypatch, tmp_path: Path) -> None:
    """Agent should forward the CLI debug flag to the LLM client."""

    captured: dict[str, object] = {}

    class FakeLLMClient:
        def __init__(
            self,
            model: str | None = None,
            debug: bool | None = None,
            tool_registry: ToolRegistry | None = None,
        ) -> None:
            captured["model"] = model
            captured["debug"] = debug
            captured["tool_registry"] = tool_registry

    class FakePromptSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr("coding_agent.agent.core.LLMClient", FakeLLMClient)
    monkeypatch.setattr("coding_agent.agent.core.PromptSession", FakePromptSession)
    monkeypatch.setattr("coding_agent.agent.core.FileHistory", lambda *args, **kwargs: object())
    monkeypatch.setattr("coding_agent.agent.core.AutoSuggestFromHistory", lambda: object())
    monkeypatch.setattr(Agent, "_init_tools", lambda self: None)

    agent = Agent(working_dir=str(tmp_path), model="gpt-4o-mini", debug=True)

    assert captured["model"] == "gpt-4o-mini"
    assert captured["debug"] is True
    assert isinstance(captured["tool_registry"], ToolRegistry)
    assert agent.llm is not None


def test_agent_configures_prompt_session_with_slash_completer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Interactive sessions should enable slash completion while typing."""

    captured: dict[str, object] = {}

    class FakeLLMClient:
        def __init__(
            self,
            model: str | None = None,
            debug: bool | None = None,
            tool_registry: ToolRegistry | None = None,
        ) -> None:
            return None

    class FakePromptSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr("coding_agent.agent.core.LLMClient", FakeLLMClient)
    monkeypatch.setattr("coding_agent.agent.core.PromptSession", FakePromptSession)
    monkeypatch.setattr("coding_agent.agent.core.FileHistory", lambda *args, **kwargs: object())
    monkeypatch.setattr("coding_agent.agent.core.AutoSuggestFromHistory", lambda: object())
    monkeypatch.setattr(Agent, "_init_tools", lambda self: None)

    Agent(working_dir=str(tmp_path), model="gpt-4o-mini", debug=False)

    prompt_kwargs = captured["kwargs"]
    assert isinstance(prompt_kwargs["completer"], SlashCommandCompleter)
    assert prompt_kwargs["complete_while_typing"] is True


def test_agent_load_system_prompt_includes_workspace_skills(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The rendered system prompt should advertise discovered workspace skills."""

    captured: dict[str, object] = {}
    skill_dir = tmp_path / ".codex" / "skills" / "reviewer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Reviewer\n\nReview code changes for regressions and missing tests.\n",
        encoding="utf-8",
    )

    class FakeLLMClient:
        def __init__(
            self,
            model: str | None = None,
            debug: bool | None = None,
            tool_registry: ToolRegistry | None = None,
        ) -> None:
            captured["tool_registry"] = tool_registry

    class FakePromptSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr("coding_agent.agent.core.LLMClient", FakeLLMClient)
    monkeypatch.setattr("coding_agent.agent.core.PromptSession", FakePromptSession)
    monkeypatch.setattr("coding_agent.agent.core.FileHistory", lambda *args, **kwargs: object())
    monkeypatch.setattr("coding_agent.agent.core.AutoSuggestFromHistory", lambda: object())
    monkeypatch.setattr(Agent, "_init_tools", lambda self: None)

    agent = Agent(working_dir=str(tmp_path), model="gpt-4o-mini", debug=False)
    prompt = agent._load_system_prompt()

    assert "## Available Skills" in prompt
    assert "Reviewer" in prompt
    assert ".codex" in prompt


def test_agent_load_system_prompt_includes_shell_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The rendered system prompt should describe the active shell environment."""

    class FakeLLMClient:
        def __init__(
            self,
            model: str | None = None,
            debug: bool | None = None,
            tool_registry: ToolRegistry | None = None,
        ) -> None:
            return None

    class FakePromptSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr("coding_agent.agent.core.LLMClient", FakeLLMClient)
    monkeypatch.setattr("coding_agent.agent.core.PromptSession", FakePromptSession)
    monkeypatch.setattr("coding_agent.agent.core.FileHistory", lambda *args, **kwargs: object())
    monkeypatch.setattr("coding_agent.agent.core.AutoSuggestFromHistory", lambda: object())
    monkeypatch.setattr(Agent, "_init_tools", lambda self: None)
    monkeypatch.setattr(
        "coding_agent.agent.core.detect_shell_profile",
        lambda: ShellProfile(
            operating_system="Windows",
            shell_name="PowerShell",
            executable="powershell",
            arguments=("-NoLogo", "-NoProfile", "-Command"),
            command_style="Use PowerShell syntax.",
            environment_syntax="Use $env:NAME='value'.",
        ),
    )

    agent = Agent(working_dir=str(tmp_path), model="gpt-4o-mini", debug=False)
    prompt = agent._load_system_prompt()

    assert "## Shell Environment" in prompt
    assert "PowerShell" in prompt
    assert "Use PowerShell syntax." in prompt


def test_init_tools_reports_mcp_startup_success(monkeypatch) -> None:
    """Startup should tell the user when MCP servers load successfully."""

    printed: list[str] = []

    agent = Agent.__new__(Agent)
    agent.debug = False
    agent.registry = ToolRegistry()
    agent.mcp_manager = type(
        "Manager",
        (),
        {"enabled": True, "server_names": ["playwright"]},
    )()

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )
    monkeypatch.setattr(Agent, "_build_base_registry", lambda self: ToolRegistry())
    monkeypatch.setattr(
        Agent,
        "_start_mcp_manager",
        lambda self, manager, registry: [
            "mcp__playwright__browser_click",
            "mcp__playwright__browser_navigate",
        ],
    )
    monkeypatch.setattr(Agent, "_sync_llm_tool_registry", lambda self: None)
    monkeypatch.setattr(Agent, "_log_loaded_tools", lambda self: None)

    Agent._init_tools(agent)

    assert any("MCP configured successfully." in line for line in printed)
    assert any("Loaded servers: `playwright`" in line for line in printed)
    assert any("Available MCP tools:" in line for line in printed)
    assert any("- mcp__playwright__browser_click" in line for line in printed)
    assert any("- mcp__playwright__browser_navigate" in line for line in printed)


def test_init_tools_shows_loading_during_mcp_startup(monkeypatch) -> None:
    """MCP startup should expose a visible loading status while connecting."""

    loading_messages: list[str] = []

    @contextmanager
    def fake_loading(self: Agent, message: str):
        loading_messages.append(message)
        yield

    agent = Agent.__new__(Agent)
    agent.debug = False
    agent.registry = ToolRegistry()
    agent.mcp_manager = type(
        "Manager",
        (),
        {"enabled": True, "server_names": ["playwright"]},
    )()

    monkeypatch.setattr(Agent, "_loading_status", fake_loading)
    monkeypatch.setattr("coding_agent.agent.core.console.print", lambda *args, **kwargs: None)
    monkeypatch.setattr(Agent, "_build_base_registry", lambda self: ToolRegistry())
    monkeypatch.setattr(
        Agent,
        "_start_mcp_manager",
        lambda self, manager, registry: ["mcp__playwright__browser_click"],
    )
    monkeypatch.setattr(Agent, "_sync_llm_tool_registry", lambda self: None)
    monkeypatch.setattr(Agent, "_log_loaded_tools", lambda self: None)

    Agent._init_tools(agent)

    assert loading_messages == ["Connecting to MCP servers: playwright..."]


def test_init_tools_reports_mcp_startup_failure(monkeypatch) -> None:
    """Startup should show a user-visible MCP failure summary before raising."""

    printed: list[str] = []

    agent = Agent.__new__(Agent)
    agent.debug = False
    agent.registry = ToolRegistry()
    agent.mcp_manager = type(
        "Manager",
        (),
        {"enabled": True, "server_names": ["playwright"]},
    )()

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )
    monkeypatch.setattr(Agent, "_build_base_registry", lambda self: ToolRegistry())

    def fail_start(self, manager: object, registry: ToolRegistry) -> list[str]:
        raise RuntimeError("playwright server exited unexpectedly")

    monkeypatch.setattr(Agent, "_start_mcp_manager", fail_start)
    monkeypatch.setattr(Agent, "_sync_llm_tool_registry", lambda self: None)
    monkeypatch.setattr(Agent, "_log_loaded_tools", lambda self: None)

    with pytest.raises(RuntimeError, match="playwright server exited unexpectedly"):
        Agent._init_tools(agent)

    assert any("MCP configuration failed." in line for line in printed)
    assert any("Configured servers: `playwright`" in line for line in printed)
    assert any("playwright server exited unexpectedly" in line for line in printed)


def test_agent_reuses_shared_event_loop_for_mcp_lifecycle(tmp_path: Path) -> None:
    """MCP startup and tool execution should run on the same asyncio runner."""

    loop_ids: list[int] = []

    class FakeManager:
        enabled = True
        server_names = ["playwright"]

        async def start(self, registry: ToolRegistry) -> list[str]:
            _ = registry
            loop_ids.append(id(asyncio.get_running_loop()))
            return ["mcp__playwright__browser_navigate"]

        async def close(self) -> None:
            return None

    class FakeTool:
        async def execute(self, **kwargs: Any) -> str:
            _ = kwargs
            loop_ids.append(id(asyncio.get_running_loop()))
            return "ok"

    agent = Agent.__new__(Agent)
    agent.debug = False
    agent.registry = ToolRegistry()
    agent.mcp_manager = FakeManager()
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent._async_runner = None

    Agent._start_mcp_manager(agent, agent.mcp_manager, agent.registry)
    agent.registry = type(
        "Registry",
        (),
        {
            "get": lambda self, name: FakeTool(),
            "list_tools": lambda self: [],
        },
    )()
    Agent._execute_tool(
        agent,
        {
            "name": "mcp__playwright__browser_navigate",
            "arguments": '{"url": "https://juejin.cn"}',
        },
    )
    Agent.close(agent)

    assert len(loop_ids) == 2
    assert loop_ids[0] == loop_ids[1]


def test_interactive_session_creates_one_log_file_on_first_submission(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The first submitted interactive message should create and announce one log file."""

    events: list[tuple[str, object]] = []

    class FakeLLMClient:
        def __init__(self) -> None:
            self.log_paths: list[Path] = []

        def set_log_path(self, log_path: Path) -> None:
            self.log_paths.append(log_path)
            events.append(("set_log_path", log_path))

        async def chat_with_tools(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> dict[str, Any]:
            events.append(("chat", messages))
            return {"content": "response", "tool_calls": []}

    printed: list[str] = []

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )
    monkeypatch.setattr(Agent, "_display_response", lambda self, content: None)
    monkeypatch.setattr("coding_agent.config.Settings.config_dir", property(lambda self: tmp_path))

    agent = Agent.__new__(Agent)
    agent.working_dir = tmp_path
    agent.model = "gpt-4o-mini"
    agent.debug = False
    agent.llm = FakeLLMClient()
    agent.history = []
    agent.registry = ToolRegistry()
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent._interaction_log_path = None

    Agent._process_message(agent, "first prompt")
    Agent._process_message(agent, "second prompt")

    assert len(agent.llm.log_paths) == 1
    log_path = agent.llm.log_paths[0]
    assert log_path.exists()
    assert log_path.parent == tmp_path / "logs"
    assert len([line for line in printed if line.startswith("[dim]LLM log file: ")]) == 1

    first_log_print_index = next(
        index for index, line in enumerate(printed) if line.startswith("[dim]LLM log file: ")
    )
    first_chat_index = next(index for index, event in enumerate(events) if event[0] == "chat")
    set_log_path_index = next(
        index for index, event in enumerate(events) if event[0] == "set_log_path"
    )

    assert set_log_path_index < first_chat_index
    assert first_log_print_index == 0


def test_process_message_streams_markdown_with_live(monkeypatch, tmp_path: Path) -> None:
    """Interactive processing should render streamed output through Rich Markdown."""

    printed: list[str] = []
    markdown_updates: list[str] = []
    live_events: list[str] = []
    loading_events: list[str] = []

    class FakeMarkdown:
        def __init__(self, content: str) -> None:
            self.content = content
            markdown_updates.append(content)

    class FakeLive:
        def __init__(
            self,
            renderable: FakeMarkdown,
            *,
            console: object,
            refresh_per_second: int,
            transient: bool,
        ) -> None:
            self.renderable = renderable
            live_events.append(f"init:{renderable.content}")

        def start(self) -> None:
            live_events.append("start")

        def update(self, renderable: FakeMarkdown, *, refresh: bool) -> None:
            self.renderable = renderable
            live_events.append(f"update:{renderable.content}")

        def stop(self) -> None:
            live_events.append("stop")

    class FakeStatus:
        def stop(self) -> None:
            loading_events.append("stop")

    class FakeLLMClient:
        def set_log_path(self, log_path: Path) -> None:
            return None

        async def chat_with_tools(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> dict[str, Any]:
            on_content_chunk = kwargs["on_content_chunk"]
            on_content_chunk("你")
            on_content_chunk("好")
            return {"content": "你好", "tool_calls": []}

    @contextmanager
    def fake_loading(self: Agent, message: str):
        loading_events.append(message)
        yield FakeStatus()

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )
    monkeypatch.setattr("coding_agent.agent.core.Markdown", FakeMarkdown)
    monkeypatch.setattr("coding_agent.agent.core.Live", FakeLive)
    monkeypatch.setattr(Agent, "_loading_status", fake_loading)
    monkeypatch.setattr(
        Agent,
        "_display_response",
        lambda self, content: printed.append(f"display:{content}"),
    )
    monkeypatch.setattr("coding_agent.config.Settings.config_dir", property(lambda self: tmp_path))

    agent = Agent.__new__(Agent)
    agent.working_dir = tmp_path
    agent.model = "gpt-4o-mini"
    agent.debug = False
    agent.llm = FakeLLMClient()
    agent.history = []
    agent.registry = ToolRegistry()
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent._interaction_log_path = None

    Agent._process_message(agent, "hello")

    assert any(line == "\n[bold blue]Agent:[/bold blue]" for line in printed)
    assert loading_events == ["Thinking...", "stop"]
    assert markdown_updates == ["你", "你好"]
    assert live_events == ["init:你", "start", "update:你好", "stop"]
    assert not any(line.startswith("display:") for line in printed)


def test_process_message_streams_reasoning_content_before_answer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Streaming reasoning_content should be shown before normal content arrives."""

    live_events: list[str] = []
    loading_events: list[str] = []

    class FakeRenderable(str):
        """Simple renderable marker for assertions."""

    class FakeLive:
        def __init__(
            self,
            renderable: FakeRenderable,
            *,
            console: object,
            refresh_per_second: int,
            transient: bool,
        ) -> None:
            _ = (console, refresh_per_second, transient)
            live_events.append(f"init:{renderable}")

        def start(self) -> None:
            live_events.append("start")

        def update(self, renderable: FakeRenderable, *, refresh: bool) -> None:
            _ = refresh
            live_events.append(f"update:{renderable}")

        def stop(self) -> None:
            live_events.append("stop")

    class FakeStatus:
        def stop(self) -> None:
            loading_events.append("stop")

    class FakeLLMClient:
        def set_log_path(self, log_path: Path) -> None:
            return None

        async def chat_with_tools(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> dict[str, Any]:
            kwargs["on_reasoning_chunk"]("先分析问题")
            kwargs["on_content_chunk"]("最终答案")
            return {
                "content": "最终答案",
                "reasoning_content": "先分析问题",
                "tool_calls": [],
            }

    @contextmanager
    def fake_loading(self: Agent, message: str):
        loading_events.append(message)
        yield FakeStatus()

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("coding_agent.agent.core.Live", FakeLive)
    monkeypatch.setattr(Agent, "_loading_status", fake_loading)
    monkeypatch.setattr(
        Agent,
        "_build_stream_renderable",
        lambda self, *, content, reasoning_content: FakeRenderable(
            f"reasoning={reasoning_content}|content={content}"
        ),
    )
    monkeypatch.setattr(Agent, "_display_response", lambda self, content: None)
    monkeypatch.setattr("coding_agent.config.Settings.config_dir", property(lambda self: tmp_path))

    agent = Agent.__new__(Agent)
    agent.working_dir = tmp_path
    agent.model = "moonshot/kimi-k2.5"
    agent.debug = False
    agent.llm = FakeLLMClient()
    agent.history = []
    agent.registry = ToolRegistry()
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent._interaction_log_path = None

    Agent._process_message(agent, "hello")

    assert loading_events == ["Thinking...", "stop"]
    assert live_events == [
        "init:reasoning=先分析问题|content=",
        "start",
        "update:reasoning=先分析问题|content=最终答案",
        "stop",
    ]


def test_process_message_preserves_reasoning_content_for_tool_calls(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Tool-call assistant messages should keep reasoning_content for Moonshot follow-ups."""

    class FakeLLMClient:
        def __init__(self) -> None:
            self.responses = [
                {
                    "content": "",
                    "reasoning_content": "need to inspect files",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "name": "list_directory",
                            "arguments": '{"path": ".", "recursive": false}',
                        }
                    ],
                },
                {
                    "content": "summary",
                    "tool_calls": [],
                },
            ]

        def set_log_path(self, log_path: Path) -> None:
            return None

        async def chat_with_tools(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> dict[str, Any]:
            return self.responses.pop(0)

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("coding_agent.config.Settings.config_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(
        Agent,
        "_execute_tool",
        lambda self, tool_call, finish_reason=None: "tool output",
    )
    monkeypatch.setattr(Agent, "_display_response", lambda self, content: None)

    agent = Agent.__new__(Agent)
    agent.working_dir = tmp_path
    agent.model = "moonshot/kimi-k2.5"
    agent.debug = False
    agent.llm = FakeLLMClient()
    agent.history = []
    agent.registry = ToolRegistry()
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent._interaction_log_path = None

    Agent._process_message(agent, "what is this project")

    assistant_messages = [message for message in agent.history if message["role"] == "assistant"]
    assert assistant_messages[0]["reasoning_content"] == "need to inspect files"


def test_execute_tool_prints_user_visible_status(monkeypatch, tmp_path: Path) -> None:
    """Tool execution should print visible start and completion hints for the user."""

    printed: list[str] = []
    loading_messages: list[str] = []

    class FakeTool:
        async def execute(self, **kwargs: Any) -> str:
            return "tool output"

    @contextmanager
    def fake_loading(self: Agent, message: str):
        loading_messages.append(message)
        yield

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )
    monkeypatch.setattr(Agent, "_loading_status", fake_loading)

    agent = Agent.__new__(Agent)
    agent.registry = type(
        "Registry",
        (),
        {
            "get": lambda self, name: FakeTool(),
            "list_tools": lambda self: [],
        },
    )()
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent.debug = False

    result = Agent._execute_tool(
        agent,
        {
            "name": "list_directory",
            "arguments": '{"path": ".", "recursive": false}',
        },
    )

    assert result == "tool output"
    assert loading_messages == ["Running tool list_directory..."]
    assert any(line.startswith("[dim]Calling tool list_directory: ") for line in printed)
    assert any(line == "[dim]Tool list_directory completed[/dim]" for line in printed)


def test_execute_tool_displays_file_change_preview(monkeypatch, tmp_path: Path) -> None:
    """Successful file edits should surface a diff preview in the CLI."""

    previews = []

    class FakeTool:
        async def execute(self, **kwargs: Any) -> str:
            file_path = Path(kwargs["ctx"]["working_dir"]) / kwargs["path"]
            file_path.write_text(kwargs["content"], encoding="utf-8")
            return f"Successfully wrote {len(kwargs['content'])} characters to '{kwargs['path']}'."

    @contextmanager
    def fake_loading(self: Agent, message: str):
        yield

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(Agent, "_loading_status", fake_loading)
    monkeypatch.setattr(
        Agent,
        "_display_file_change_preview",
        lambda self, preview: previews.append(preview),
    )

    agent = Agent.__new__(Agent)
    agent.registry = type(
        "Registry",
        (),
        {
            "get": lambda self, name: FakeTool(),
            "list_tools": lambda self: [],
        },
    )()
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent.debug = False

    result = Agent._execute_tool(
        agent,
        {
            "name": "write_file",
            "arguments": '{"path": "demo.py", "content": "print(\\"hi\\")\\n"}',
        },
    )

    assert result == "Successfully wrote 12 characters to 'demo.py'."
    assert len(previews) == 1
    assert previews[0].operation == "created"
    assert previews[0].path == "demo.py"
    assert previews[0].added_lines == 1


def test_execute_tool_returns_retry_guidance_for_truncated_arguments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Truncated tool-call JSON should return a recovery hint instead of a raw parse failure."""

    class FakeTool:
        async def execute(self, **kwargs: Any) -> str:
            return "tool output"

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: None,
    )

    agent = Agent.__new__(Agent)
    agent.registry = type(
        "Registry",
        (),
        {
            "get": lambda self, name: FakeTool(),
            "list_tools": lambda self: [],
        },
    )()
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent.debug = False

    result = Agent._execute_tool(
        agent,
        {
            "name": "write_file",
            "arguments": '{"path": "demo.py", "content": "unterminated',
        },
        finish_reason="length",
    )

    assert "Tool arguments were truncated" in result
    assert "write_file" in result
    assert "append_file" in result


def test_process_message_handles_claude_truncated_tool_arguments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Anthropic-format tool_use messages should stay valid even when args are truncated."""

    class FakeLLMClient:
        def __init__(self) -> None:
            self.responses = [
                {
                    "content": "",
                    "finish_reason": "length",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "name": "write_file",
                            "arguments": '{"path": "demo.py", "content": "unterminated',
                        }
                    ],
                },
                {
                    "content": "retry with chunks",
                    "tool_calls": [],
                },
            ]

        def set_log_path(self, log_path: Path) -> None:
            return None

        async def chat_with_tools(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> dict[str, Any]:
            return self.responses.pop(0)

    class FakeTool:
        async def execute(self, **kwargs: Any) -> str:
            return "tool output"

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("coding_agent.config.Settings.config_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(Agent, "_display_response", lambda self, content: None)

    agent = Agent.__new__(Agent)
    agent.working_dir = tmp_path
    agent.model = "claude-3-7-sonnet"
    agent.debug = False
    agent.llm = FakeLLMClient()
    agent.history = []
    agent.registry = type(
        "Registry",
        (),
        {
            "get": lambda self, name: FakeTool(),
            "list_tools": lambda self: [],
        },
    )()
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent._interaction_log_path = None

    Agent._process_message(agent, "create a file")

    assistant_message = next(message for message in agent.history if message["role"] == "assistant")
    tool_use_block = assistant_message["content"][1]

    assert tool_use_block["type"] == "tool_use"
    assert tool_use_block["input"] == {"_invalid_tool_arguments": True}

    tool_result = next(
        message
        for message in agent.history
        if message["role"] == "user" and isinstance(message["content"], list)
    )
    assert "Tool arguments were truncated" in tool_result["content"][0]["content"]


def test_run_interactive_handles_slash_commands_locally(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Slash commands should bypass the LLM loop in interactive sessions."""

    prompts = iter(["/mcp", "quit"])
    displayed: list[str] = []
    processed: list[str] = []

    class FakeSession:
        def prompt(self, *args: object, **kwargs: object) -> str:
            return next(prompts)

    class FakeSlashCommands:
        def execute(self, user_input: str, *, ctx: object) -> SlashCommandResult | None:
            if user_input == "/mcp":
                return SlashCommandResult(output="mcp result")
            return None

    agent = Agent.__new__(Agent)
    agent.session = FakeSession()
    agent.slash_commands = FakeSlashCommands()
    agent.working_dir = tmp_path
    agent.skill_catalog = SkillCatalog()
    agent.skill_manager = type("Manager", (), {"discover": lambda self: SkillCatalog()})()

    monkeypatch.setattr("coding_agent.agent.core.console.print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Agent,
        "_display_slash_command_result",
        lambda self, content: displayed.append(content),
    )
    monkeypatch.setattr(
        Agent, "_process_message", lambda self, user_input: processed.append(user_input)
    )

    Agent.run_interactive(agent)

    assert displayed == ["mcp result"]
    assert processed == []


def test_dispatch_slash_command_shows_loading_for_mcp_updates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """MCP-mutating slash commands should run inside a visible loading state."""

    loading_messages: list[str] = []
    seen_inputs: list[str] = []

    class FakeSlashCommands:
        def execute(self, user_input: str, *, ctx: object) -> SlashCommandResult | None:
            _ = ctx
            seen_inputs.append(user_input)
            return SlashCommandResult(output="reloaded")

    @contextmanager
    def fake_loading(self: Agent, message: str):
        loading_messages.append(message)
        yield

    agent = Agent.__new__(Agent)
    agent.slash_commands = FakeSlashCommands()
    agent.working_dir = tmp_path
    agent.skill_catalog = SkillCatalog()
    agent.skill_manager = type("Manager", (), {"discover": lambda self: SkillCatalog()})()

    settings_obj = Settings(_env_file=None)
    monkeypatch.setattr("coding_agent.agent.core.settings", settings_obj)
    monkeypatch.setattr(Agent, "_loading_status", fake_loading)

    result = Agent._dispatch_slash_command(agent, "/mcp reload")

    assert result is not None
    assert result.output == "reloaded"
    assert seen_inputs == ["/mcp reload"]
    assert loading_messages == ["Reloading MCP tools..."]


def test_run_once_returns_local_slash_command_output(tmp_path: Path) -> None:
    """Single-shot runs should resolve slash commands without calling the LLM."""

    class FakeSlashCommands:
        def execute(self, user_input: str, *, ctx: object) -> SlashCommandResult | None:
            if user_input == "/mcp":
                return SlashCommandResult(output="mcp result")
            return None

    agent = Agent.__new__(Agent)
    agent.slash_commands = FakeSlashCommands()
    agent.working_dir = tmp_path
    agent.history = []
    agent.mcp_manager = type("Manager", (), {"enabled": False})()
    agent.skill_catalog = SkillCatalog()
    agent.skill_manager = type("Manager", (), {"discover": lambda self: SkillCatalog()})()

    result = Agent.run_once(agent, "/mcp")

    assert result == "mcp result"
    assert agent.history == []
