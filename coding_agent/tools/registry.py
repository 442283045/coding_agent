"""Tool registry primitives for local and dynamic agent tools."""

import inspect
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""

    name: str
    type: str
    description: str
    required: bool = True


class Tool:
    """A tool that can be called by the agent."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Awaitable[Any]],
        parameters: list[ToolParameter] | None = None,
        *,
        input_schema: Mapping[str, Any] | None = None,
        source: Literal["local", "mcp"] = "local",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or []
        self.input_schema = dict(input_schema) if input_schema is not None else None
        self.source = source
        self.metadata = dict(metadata) if metadata is not None else {}

    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given arguments."""
        result = await self.func(**kwargs)
        return str(result) if result is not None else ""

    def copy(self) -> Tool:
        """Create a detached copy of the tool definition."""
        return Tool(
            name=self.name,
            description=self.description,
            func=self.func,
            parameters=[parameter.model_copy(deep=True) for parameter in self.parameters],
            input_schema=deepcopy(self.input_schema),
            source=self.source,
            metadata=deepcopy(self.metadata),
        )

    def _build_input_schema(self) -> dict[str, Any]:
        """Return the JSON schema used by provider tool APIs."""
        if self.input_schema is not None:
            return deepcopy(self.input_schema)

        return {
            "type": "object",
            "properties": {
                parameter.name: {
                    "type": parameter.type,
                    "description": parameter.description,
                }
                for parameter in self.parameters
            },
            "required": [parameter.name for parameter in self.parameters if parameter.required],
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._build_input_schema(),
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._build_input_schema(),
        }


def get_type_schema(annotation: Any) -> str:
    """Get JSON schema type from Python annotation."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if origin is list:
            return "array"
        if origin is dict:
            return "object"
        if origin in {tuple, set, frozenset}:
            return "array"
        if str(origin) in {"typing.Union", "types.UnionType"}:
            for arg in args:
                if arg is not type(None):
                    return get_type_schema(arg)

    return type_map.get(annotation, "string")


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool if it exists."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_openai_tools(self) -> list[dict[str, Any]]:
        """List tools in OpenAI format."""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def list_anthropic_tools(self) -> list[dict[str, Any]]:
        """List tools in Anthropic format."""
        return [tool.to_anthropic_format() for tool in self._tools.values()]

    def copy(self) -> ToolRegistry:
        """Create a registry copy with detached tool definitions."""
        copied = ToolRegistry()
        for tool in self._tools.values():
            copied.register(tool.copy())
        return copied

    def tool(
        self,
        name: str,
        description: str,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator to register a function as a tool."""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            sig = inspect.signature(func)
            params: list[ToolParameter] = []

            for param_name, param in sig.parameters.items():
                # Skip 'ctx' parameter (injected context)
                if param_name == "ctx":
                    continue

                param_type = get_type_schema(param.annotation)
                is_required = param.default is inspect.Parameter.empty
                default_desc = f" (default: {param.default})" if not is_required else ""

                params.append(
                    ToolParameter(
                        name=param_name,
                        type=param_type,
                        description=f"Parameter {param_name}{default_desc}",
                        required=is_required,
                    )
                )

            tool = Tool(name, description, func, params)
            self.register(tool)
            return func

        return decorator


# Global registry instance
registry = ToolRegistry()
