"""Tests for the core agent loop."""

from pathlib import Path

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
