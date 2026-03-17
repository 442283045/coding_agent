"""Tool registry for managing agent tools."""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

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
        parameters: list[ToolParameter],
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters

    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given arguments."""
        result = await self.func(**kwargs)
        return str(result) if result is not None else ""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p.name: {"type": p.type, "description": p.description}
                        for p in self.parameters
                    },
                    "required": [p.name for p in self.parameters if p.required],
                },
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    p.name: {"type": p.type, "description": p.description} for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
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

    # Handle Optional types
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        if origin is list or origin is list:
            return "array"
        if origin is dict:
            return "object"
        # Handle Union types (including Optional)
        if origin is type or str(getattr(origin, "__name__", "")) == "Union":
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
