"""Tests for LLM client request/response logging."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from coding_agent.llm.client import LLMClient
from coding_agent.llm.runtime import configure_litellm_runtime_env
from coding_agent.tools.registry import Tool, ToolRegistry


def _build_response(
    *,
    content: str,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> SimpleNamespace:
    """Create a LiteLLM-like response object for tests."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls or [],
                ),
            )
        ]
    )


@pytest.mark.asyncio
async def test_chat_with_tools_logs_formatted_request_and_response(monkeypatch) -> None:
    """Debug mode should print formatted LLM request and response logs."""
    printed: list[str] = []

    def capture_print(*args: object, **kwargs: object) -> None:
        printed.append(" ".join(str(arg) for arg in args))

    async def fake_acompletion(**kwargs: Any) -> Any:
        return _build_response(
            content="Hello from the model",
            tool_calls=[
                SimpleNamespace(
                    id="tool-1",
                    function=SimpleNamespace(
                        name="read_file",
                        arguments='{"path": "README.md"}',
                    ),
                )
            ],
        )

    monkeypatch.setattr("coding_agent.llm.client.console.print", capture_print)
    monkeypatch.setattr("coding_agent.llm.client.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr(
        LLMClient,
        "_get_tools_format",
        lambda self: [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    client = LLMClient(model="gpt-4o-mini", debug=True)

    first = await client.chat_with_tools([{"role": "user", "content": "你好"}], stream=False)
    second = await client.chat_with_tools([{"role": "user", "content": "再说一次"}], stream=False)

    output = "\n".join(printed)

    assert first["content"] == "Hello from the model"
    assert first["finish_reason"] == "stop"
    assert second["tool_calls"][0]["name"] == "read_file"
    assert output.count("Agent Run Log - ") == 1
    assert "[1] REQUEST" in output
    assert "[2] RESPONSE" in output
    assert "[3] REQUEST" in output
    assert "[4] RESPONSE" in output
    assert '"content": "你好"' in output
    assert '"content": "再说一次"' in output
    assert '"tools": [' in output
    assert '"read_file"' in output
    assert '"finish_reason": "stop"' in output
    assert '"name": "read_file"' in output


@pytest.mark.asyncio
async def test_chat_with_tools_skips_llm_logs_without_debug(monkeypatch) -> None:
    """Non-debug mode should not print formatted LLM logs."""
    printed: list[str] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        return _build_response(content="quiet")

    monkeypatch.setattr(
        "coding_agent.llm.client.console.print",
        lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )
    monkeypatch.setattr("coding_agent.llm.client.litellm.acompletion", fake_acompletion)

    client = LLMClient(model="gpt-4o-mini", debug=False)
    result = await client.chat_with_tools([{"role": "user", "content": "hello"}], stream=False)

    assert result["content"] == "quiet"
    assert printed == []


def test_llm_client_normalizes_kimi_model_and_sets_moonshot_api_key(monkeypatch) -> None:
    """Moonshot shorthand models should be normalized and use the configured API key."""
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_BASE", raising=False)
    monkeypatch.setattr("coding_agent.llm.client.settings.moonshot_api_key", "moonshot-test-key")
    monkeypatch.setattr(
        "coding_agent.llm.client.settings.moonshot_api_base",
        "https://api.moonshot.cn/v1",
    )

    client = LLMClient(model="kimi-k2.5", debug=False)

    assert client.model == "moonshot/kimi-k2.5"
    assert os.environ["MOONSHOT_API_KEY"] == "moonshot-test-key"
    assert os.environ["MOONSHOT_API_BASE"] == "https://api.moonshot.cn/v1"


def test_llm_client_uses_supported_temperature_for_kimi_k25() -> None:
    """Kimi K2.5 requests should use the server-supported temperature value."""
    client = LLMClient(model="moonshot/kimi-k2.5", debug=False)

    request = client._build_completion_request(
        [{"role": "user", "content": "hello"}],
        stream=False,
    )

    assert request["temperature"] == 1.0


def test_configure_litellm_runtime_env_forces_local_model_cost_map(monkeypatch) -> None:
    """LiteLLM should be forced to use the bundled local model cost map."""
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)

    configure_litellm_runtime_env()

    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "true"


def test_handle_response_preserves_reasoning_content() -> None:
    """Non-streaming responses should retain reasoning_content when present."""
    client = LLMClient(model="gpt-4o-mini", debug=False)

    result = client._handle_response(
        _build_response(
            content="done",
            reasoning_content="thought process",
        )
    )

    assert result["content"] == "done"
    assert result["finish_reason"] == "stop"
    assert result["reasoning_content"] == "thought process"


def test_llm_client_uses_instance_tool_registry() -> None:
    """Each LLM client should read tools from its injected registry."""

    async def execute_tool(**kwargs: object) -> str:
        return "ok"

    tool_registry = ToolRegistry()
    tool_registry.register(
        Tool(
            name="mcp__filesystem__read_file",
            description="Read from MCP filesystem",
            func=execute_tool,
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            source="mcp",
        )
    )

    client = LLMClient(model="gpt-4o-mini", debug=False, tool_registry=tool_registry)
    request = client._build_completion_request(
        [{"role": "user", "content": "hello"}],
        stream=False,
    )

    assert request["tools"][0]["function"]["name"] == "mcp__filesystem__read_file"


@pytest.mark.asyncio
async def test_chat_with_tools_writes_logs_to_file(monkeypatch, tmp_path: Path) -> None:
    """LLM logs should be persisted once a log file path is configured."""
    printed: list[str] = []
    log_path = tmp_path / "session.log"

    async def fake_acompletion(**kwargs: Any) -> Any:
        return _build_response(content="Logged to disk")

    monkeypatch.setattr(
        "coding_agent.llm.client.console.print",
        lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )
    monkeypatch.setattr("coding_agent.llm.client.litellm.acompletion", fake_acompletion)

    client = LLMClient(model="gpt-4o-mini", debug=False)
    client.set_log_path(log_path)

    result = await client.chat_with_tools(
        [{"role": "user", "content": "write this to a file"}],
        stream=False,
    )

    output = log_path.read_text(encoding="utf-8")

    assert result["content"] == "Logged to disk"
    assert printed == []
    assert "Agent Run Log - " in output
    assert "[1] REQUEST" in output
    assert "[2] RESPONSE" in output
    assert '"content": "write this to a file"' in output
    assert '"content": "Logged to disk"' in output


@pytest.mark.asyncio
async def test_chat_with_tools_streams_by_default_and_collects_tool_calls(monkeypatch) -> None:
    """Streaming should be the default path for chat_with_tools calls."""
    captured_request: dict[str, Any] = {}
    chunks_seen: list[str] = []
    reasoning_chunks_seen: list[str] = []

    async def fake_stream() -> Any:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content="Hel",
                        reasoning_content="Think ",
                        tool_calls=None,
                    ),
                )
            ]
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content="lo",
                        reasoning_content="more",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="tool-1",
                                function=SimpleNamespace(
                                    name="read_file",
                                    arguments='{"path":"REA',
                                ),
                            )
                        ],
                    ),
                )
            ]
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                function=SimpleNamespace(name=None, arguments='DME.md"}'),
                            )
                        ],
                    ),
                )
            ]
        )

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured_request.update(kwargs)
        return fake_stream()

    monkeypatch.setattr("coding_agent.llm.client.litellm.acompletion", fake_acompletion)

    client = LLMClient(model="gpt-4o-mini", debug=False)
    result = await client.chat_with_tools(
        [{"role": "user", "content": "hello"}],
        on_content_chunk=chunks_seen.append,
        on_reasoning_chunk=reasoning_chunks_seen.append,
    )

    assert captured_request["stream"] is True
    assert chunks_seen == ["Hel", "lo"]
    assert reasoning_chunks_seen == ["Think ", "more"]
    assert result["content"] == "Hello"
    assert result["finish_reason"] == "tool_calls"
    assert result["reasoning_content"] == "Think more"
    assert result["tool_calls"] == [
        {
            "id": "tool-1",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        }
    ]
