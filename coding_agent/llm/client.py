"""Unified LLM client supporting multiple providers."""

import json
from typing import Any, AsyncIterator

import litellm
from rich.console import Console

from coding_agent.config import settings
from coding_agent.tools.registry import registry

console = Console()


class LLMClient:
    """Unified client for LLM interactions."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.default_model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature

        # Configure litellm
        litellm.set_verbose = settings.debug

        # Set API keys
        if settings.openai_api_key:
            litellm.openai_key = settings.openai_api_key
        if settings.anthropic_api_key:
            litellm.anthropic_key = settings.anthropic_api_key

    def _get_tools_format(self) -> list[dict[str, Any]]:
        """Get tools in the appropriate format for the model."""
        # Detect model provider
        model_lower = self.model.lower()

        if "claude" in model_lower:
            return registry.list_anthropic_tools()
        else:
            # OpenAI-compatible format
            return registry.list_openai_tools()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        stream: bool = True,
    ) -> dict[str, Any] | AsyncIterator[str]:
        """Send a chat completion request."""
        tools = self._get_tools_format()

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=stream,
            )

            if stream:
                return self._handle_stream(response)
            else:
                return self._handle_response(response)

        except Exception as e:
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

        # Extract tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                result["tool_calls"].append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )

        return result

    async def _handle_stream(self, response: Any) -> AsyncIterator[str]:
        """Handle streaming response."""
        async for chunk in response:
            delta = chunk.choices[0].delta

            if delta.content:
                yield delta.content

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send a chat request and return structured response with tool calls."""
        tools = self._get_tools_format()

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False,
            )

            return self._handle_response(response)

        except Exception as e:
            console.print(f"[red]LLM Error: {e}[/red]")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximation)."""
        # Use tiktoken if available, otherwise rough estimate
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model("gpt-4")
            return len(enc.encode(text))
        except Exception:
            # Rough estimate: ~4 chars per token
            return len(text) // 4

    def format_messages_for_provider(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format messages according to provider requirements."""
        # LiteLLM handles most of this, but we can add custom formatting here
        return messages
