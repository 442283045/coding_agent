"""Unified LLM client supporting multiple providers."""

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import litellm
from rich.console import Console

from coding_agent.config import normalize_model_name, settings
from coding_agent.tools.registry import registry

console = Console()


def _format_timestamp(value: datetime, *, include_milliseconds: bool) -> str:
    """Format timestamps for human-readable LLM logs."""
    if include_milliseconds:
        return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return value.strftime("%Y-%m-%d %H:%M:%S")


@dataclass(slots=True)
class _LLMInteractionLogger:
    """Pretty-printer for structured LLM request and response logs."""

    enabled: bool
    _entry_index: int = 1
    _header_printed: bool = False
    _run_started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def log_request(self, payload: Mapping[str, Any]) -> None:
        """Print a formatted request log entry."""
        self._log_entry("REQUEST", "LLM Request:", payload)

    def log_response(self, payload: Mapping[str, Any]) -> None:
        """Print a formatted response log entry."""
        self._log_entry("RESPONSE", "LLM Response:", payload)

    def log_error(self, error: Exception) -> None:
        """Print a formatted error log entry."""
        self._log_entry(
            "ERROR",
            "LLM Error:",
            {
                "type": type(error).__name__,
                "message": str(error),
            },
        )

    def _log_entry(
        self,
        label: str,
        title: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Render a single log section."""
        if not self.enabled:
            return

        if not self._header_printed:
            self._print_header()

        timestamp = datetime.now().astimezone()
        entry_index = self._entry_index
        self._entry_index += 1

        lines = [
            "",
            "-" * 80,
            f"[{entry_index}] {label}",
            f"Timestamp: {_format_timestamp(timestamp, include_milliseconds=True)}",
            "-" * 80,
            title,
            "",
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            "",
        ]
        console.print("\n".join(lines), markup=False, highlight=False, soft_wrap=True)

    def _print_header(self) -> None:
        """Render the log header once per client instance."""
        lines = [
            "=" * 80,
            (
                "Agent Run Log - "
                f"{_format_timestamp(self._run_started_at, include_milliseconds=False)}"
            ),
            "=" * 80,
            "",
        ]
        console.print("\n".join(lines), markup=False, highlight=False, soft_wrap=True)
        self._header_printed = True


class LLMClient:
    """Unified client for LLM interactions."""

    def __init__(self, model: str | None = None, debug: bool | None = None):
        self.model = normalize_model_name(model or settings.default_model)
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.debug = settings.debug if debug is None else debug
        self._logger = _LLMInteractionLogger(enabled=self.debug)

        # Configure litellm
        litellm.set_verbose = self.debug  # type: ignore[attr-defined]

        # Set API keys
        if settings.openai_api_key:
            litellm.openai_key = settings.openai_api_key
        if settings.anthropic_api_key:
            litellm.anthropic_key = settings.anthropic_api_key
        if settings.moonshot_api_key:
            os.environ.setdefault("MOONSHOT_API_KEY", settings.moonshot_api_key)

    def _get_tools_format(self) -> list[dict[str, Any]]:
        """Get tools in the appropriate format for the model."""
        model_lower = self.model.lower()

        if "claude" in model_lower:
            return registry.list_anthropic_tools()
        return registry.list_openai_tools()

    def _build_completion_request(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build a LiteLLM completion request."""
        tools = self._get_tools_format()
        return {
            "model": self.model,
            "messages": messages,
            "tools": tools if tools else None,
            "tool_choice": "auto" if tools else None,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }

    def _build_request_log_payload(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Create a compact log-friendly view of the outgoing request."""
        payload: dict[str, Any] = {
            "messages": request["messages"],
            "model": request["model"],
            "max_tokens": request["max_tokens"],
            "temperature": request["temperature"],
        }
        tool_names = self._extract_tool_names(request.get("tools"))
        if tool_names:
            payload["tools"] = tool_names
        if request.get("stream"):
            payload["stream"] = True
        return payload

    def _extract_tool_names(self, tools: object) -> list[str]:
        """Extract tool names from OpenAI or Anthropic tool schemas."""
        if not isinstance(tools, Sequence):
            return []

        names: list[str] = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue

            match tool:
                case {"name": str(name)}:
                    names.append(name)
                case {"function": {"name": str(name)}}:
                    names.append(name)

        return names

    def _build_response_log_payload(
        self,
        response: Any,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create a structured response payload for logging."""
        payload = dict(result)
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            payload["finish_reason"] = finish_reason
        return payload

    async def chat(
        self,
        messages: list[dict[str, Any]],
        stream: bool = True,
    ) -> dict[str, Any] | AsyncIterator[str]:
        """Send a chat completion request."""
        request = self._build_completion_request(messages, stream=stream)
        self._logger.log_request(self._build_request_log_payload(request))

        try:
            response = await litellm.acompletion(**request)

            if stream:
                return self._handle_stream(response)

            result = self._handle_response(response)
            self._logger.log_response(self._build_response_log_payload(response, result))
            return result

        except Exception as e:
            self._logger.log_error(e)
            console.print(f"[red]LLM Error: {e}[/red]")
            raise

    def _handle_response(self, response: Any) -> dict[str, Any]:
        """Handle non-streaming response."""
        choice = response.choices[0]
        message = choice.message

        result: dict[str, Any] = {
            "content": message.content or "",
            "tool_calls": [],
        }

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                result["tool_calls"].append(
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )

        return result

    async def _handle_stream(self, response: Any) -> AsyncIterator[str]:
        """Handle streaming response."""
        chunks: list[str] = []

        try:
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
                    yield delta.content
        finally:
            self._logger.log_response(
                {
                    "content": "".join(chunks),
                    "finish_reason": "stream",
                }
            )

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send a chat request and return structured response with tool calls."""
        request = self._build_completion_request(messages, stream=False)
        self._logger.log_request(self._build_request_log_payload(request))

        try:
            response = await litellm.acompletion(**request)
            result = self._handle_response(response)
            self._logger.log_response(self._build_response_log_payload(response, result))
            return result

        except Exception as e:
            self._logger.log_error(e)
            console.print(f"[red]LLM Error: {e}[/red]")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximation)."""
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model("gpt-4")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4

    def format_messages_for_provider(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format messages according to provider requirements."""
        return messages
