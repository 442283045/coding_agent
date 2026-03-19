"""Tests for the core agent loop."""

from pathlib import Path
from typing import Any

from coding_agent.agent.core import PROMPT_STYLE, Agent


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
        def __init__(self, model: str | None = None, debug: bool | None = None) -> None:
            captured["model"] = model
            captured["debug"] = debug

    class FakePromptSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr("coding_agent.agent.core.LLMClient", FakeLLMClient)
    monkeypatch.setattr("coding_agent.agent.core.PromptSession", FakePromptSession)
    monkeypatch.setattr("coding_agent.agent.core.FileHistory", lambda *args, **kwargs: object())
    monkeypatch.setattr("coding_agent.agent.core.AutoSuggestFromHistory", lambda: object())
    monkeypatch.setattr(Agent, "_init_tools", lambda self: None)

    agent = Agent(working_dir=str(tmp_path), model="gpt-4o-mini", debug=True)

    assert captured == {"model": "gpt-4o-mini", "debug": True}
    assert agent.llm is not None


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


def test_process_message_streams_content_without_panel(monkeypatch, tmp_path: Path) -> None:
    """Interactive processing should print streamed chunks as they arrive."""

    printed: list[str] = []
    display_calls: list[str] = []

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

    monkeypatch.setattr(
        "coding_agent.agent.core.console.print",
        lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )
    monkeypatch.setattr(
        Agent,
        "_display_response",
        lambda self, content: display_calls.append(content),
    )
    monkeypatch.setattr("coding_agent.config.Settings.config_dir", property(lambda self: tmp_path))

    agent = Agent.__new__(Agent)
    agent.working_dir = tmp_path
    agent.model = "gpt-4o-mini"
    agent.debug = False
    agent.llm = FakeLLMClient()
    agent.history = []
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent._interaction_log_path = None

    Agent._process_message(agent, "hello")

    assert any(line == "\n[bold blue]Agent:[/bold blue]" for line in printed)
    assert "你" in printed
    assert "好" in printed
    assert display_calls == []


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
    monkeypatch.setattr(Agent, "_execute_tool", lambda self, tool_call: "tool output")
    monkeypatch.setattr(Agent, "_display_response", lambda self, content: None)

    agent = Agent.__new__(Agent)
    agent.working_dir = tmp_path
    agent.model = "moonshot/kimi-k2.5"
    agent.debug = False
    agent.llm = FakeLLMClient()
    agent.history = []
    agent.tool_context = {"working_dir": str(tmp_path), "debug": False}
    agent._interaction_log_path = None

    Agent._process_message(agent, "what is this project")

    assistant_messages = [message for message in agent.history if message["role"] == "assistant"]
    assert assistant_messages[0]["reasoning_content"] == "need to inspect files"
