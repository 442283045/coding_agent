"""Unified LLM client supporting multiple providers."""

import json
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from coding_agent.config import normalize_model_name, settings
from coding_agent.llm.runtime import litellm
from coding_agent.tools.registry import ToolRegistry, registry

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
    log_path: Path | None = None
    _entry_index: int = 1
    _console_header_printed: bool = False
    _file_header_written: bool = False
    _run_started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def set_log_path(self, log_path: Path) -> None:
        """Enable appending future log entries to a file."""
        self.log_path = log_path.resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.touch(exist_ok=True)
        self._file_header_written = self.log_path.stat().st_size > 0

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
        if not self.enabled and self.log_path is None:
            return

        self._ensure_header_written()

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
        self._emit("\n".join(lines))

    def _ensure_header_written(self) -> None:
        """Render the log header once per output target."""
        header = self._build_header()
        if self.enabled and not self._console_header_printed:
            console.print(header, markup=False, highlight=False, soft_wrap=True)
            self._console_header_printed = True
        if self.log_path is not None and not self._file_header_written:
            self._append_to_file(header)
            self._file_header_written = True

    def _build_header(self) -> str:
        """Build the shared header for a log stream."""
        lines = [
            "=" * 80,
            (
                "Agent Run Log - "
                f"{_format_timestamp(self._run_started_at, include_milliseconds=False)}"
            ),
            "=" * 80,
            "",
        ]
        return "\n".join(lines)

    def _emit(self, content: str) -> None:
        """Write a log entry to all configured outputs."""
        if self.enabled:
            console.print(content, markup=False, highlight=False, soft_wrap=True)
        if self.log_path is not None:
            self._append_to_file(content)

    def _append_to_file(self, content: str) -> None:
        """Append content to the configured log file."""
        if self.log_path is None:
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(content)


class LLMClient:
    """Unified client for LLM interactions."""

    def __init__(
        self,
        model: str | None = None,
        debug: bool | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.model = normalize_model_name(model or settings.default_model)
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.debug = settings.debug if debug is None else debug
        self.tool_registry = tool_registry or registry
        self._logger = _LLMInteractionLogger(enabled=self.debug)

        # Configure litellm
        litellm.set_verbose = self.debug

        # Set API keys
        if settings.openai_api_key:
            litellm.openai_key = settings.openai_api_key
        if settings.anthropic_api_key:
            litellm.anthropic_key = settings.anthropic_api_key
        if settings.moonshot_api_key:
            os.environ["MOONSHOT_API_KEY"] = settings.moonshot_api_key
        if settings.moonshot_api_base:
            os.environ["MOONSHOT_API_BASE"] = settings.moonshot_api_base

    def set_log_path(self, log_path: Path) -> None:
        """Persist future LLM interaction logs to a file."""
        self._logger.set_log_path(log_path)

    def _get_tools_format(self) -> list[dict[str, Any]]:
        """Get tools in the appropriate format for the model."""
        model_lower = self.model.lower()

        if "claude" in model_lower:
            return self.tool_registry.list_anthropic_tools()
        return self.tool_registry.list_openai_tools()

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
            "temperature": self._get_request_temperature(),
            "stream": stream,
        }

    def _get_request_temperature(self) -> float:
        """Return the effective temperature for the current model."""
        if self.model.lower() == "moonshot/kimi-k2.5":
            return 1.0
        return self.temperature

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

    def _build_stream_response_log_payload(
        self,
        result: Mapping[str, Any],
        *,
        finish_reason: str | None,
    ) -> dict[str, Any]:
        """Create a structured response payload for streaming responses."""
        payload = dict(result)
        if finish_reason is not None:
            payload["finish_reason"] = finish_reason
        else:
            payload["finish_reason"] = "stream"
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
        finish_reason = getattr(choice, "finish_reason", None)
        if isinstance(finish_reason, str) and finish_reason:
            result["finish_reason"] = finish_reason
        reasoning_content = getattr(message, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content:
            result["reasoning_content"] = reasoning_content

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

    async def _handle_stream_with_tools(
        self,
        response: Any,
        *,
        on_content_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Handle a streaming response that may contain tool calls."""
        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        tool_calls_by_index: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None

        async for chunk in response:
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason

            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                content_chunks.append(content)
                if on_content_chunk is not None:
                    on_content_chunk(content)

            reasoning_content = getattr(delta, "reasoning_content", None)
            if isinstance(reasoning_content, str) and reasoning_content:
                reasoning_chunks.append(reasoning_content)

            delta_tool_calls = getattr(delta, "tool_calls", None)
            if not delta_tool_calls:
                continue

            for delta_tool_call in delta_tool_calls:
                index = getattr(delta_tool_call, "index", 0)
                tool_call = tool_calls_by_index.setdefault(
                    index,
                    {"id": "", "name": "", "arguments": ""},
                )

                tool_call_id = getattr(delta_tool_call, "id", None)
                if isinstance(tool_call_id, str) and tool_call_id:
                    tool_call["id"] = tool_call_id

                function = getattr(delta_tool_call, "function", None)
                if function is None:
                    continue

                function_name = getattr(function, "name", None)
                if isinstance(function_name, str) and function_name:
                    tool_call["name"] = function_name

                function_arguments = getattr(function, "arguments", None)
                if isinstance(function_arguments, str) and function_arguments:
                    tool_call["arguments"] += function_arguments

        result: dict[str, Any] = {
            "content": "".join(content_chunks),
            "tool_calls": [
                tool_calls_by_index[index]
                for index in sorted(tool_calls_by_index)
                if tool_calls_by_index[index]["name"]
            ],
        }
        if isinstance(finish_reason, str) and finish_reason:
            result["finish_reason"] = finish_reason
        if reasoning_chunks:
            result["reasoning_content"] = "".join(reasoning_chunks)
        self._logger.log_response(
            self._build_stream_response_log_payload(
                result,
                finish_reason=finish_reason,
            )
        )
        return result

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = True,
        on_content_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Send a chat request and return structured response with tool calls."""
        request = self._build_completion_request(messages, stream=stream)
        self._logger.log_request(self._build_request_log_payload(request))

        try:
            response = await litellm.acompletion(**request)
            if stream:
                return await self._handle_stream_with_tools(
                    response,
                    on_content_chunk=on_content_chunk,
                )

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
