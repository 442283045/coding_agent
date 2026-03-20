"""MCP client lifecycle management and tool adaptation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from coding_agent.config import MCPServerConfig
from coding_agent.tools.registry import Tool, ToolRegistry


@dataclass(slots=True)
class MCPToolBinding:
    """Metadata for a dynamically registered MCP tool."""

    server_name: str
    remote_name: str
    tool_name: str


class _MCPClientProtocol(Protocol):
    """Typed subset of the FastMCP client used by the agent."""

    async def __aenter__(self) -> Any: ...

    async def list_tools(self) -> list[Any]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...

    async def close(self) -> None: ...


class MCPManager:
    """Connect to configured MCP servers and expose their tools through the registry."""

    def __init__(self, server_configs: Mapping[str, MCPServerConfig]) -> None:
        self._server_configs = dict(server_configs)
        self._clients: dict[str, Any] = {}
        self._tool_bindings: dict[str, MCPToolBinding] = {}

    @property
    def enabled(self) -> bool:
        """Whether any MCP servers are configured."""
        return bool(self._server_configs)

    async def start(self, registry: ToolRegistry) -> list[str]:
        """Connect to configured servers and register their tools."""
        if not self._server_configs:
            return []

        try:
            from fastmcp import Client
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MCP servers are configured, but the 'fastmcp' package is not installed."
            ) from exc

        registered_names: list[str] = []
        client_factory: Any = Client

        for server_name, server_config in self._server_configs.items():
            client = cast(_MCPClientProtocol, client_factory(server_config.to_fastmcp_config()))
            await client.__aenter__()
            self._clients[server_name] = client

            for remote_tool in await client.list_tools():
                tool_name = self._build_tool_name(server_name, remote_tool.name)
                registry.register(
                    Tool(
                        name=tool_name,
                        description=remote_tool.description or f"MCP tool from {server_name}",
                        func=self._build_executor(server_name, remote_tool.name),
                        input_schema=remote_tool.inputSchema,
                        source="mcp",
                        metadata={
                            "server_name": server_name,
                            "remote_name": remote_tool.name,
                            "title": getattr(remote_tool, "title", None),
                        },
                    )
                )
                self._tool_bindings[tool_name] = MCPToolBinding(
                    server_name=server_name,
                    remote_name=remote_tool.name,
                    tool_name=tool_name,
                )
                registered_names.append(tool_name)

        return registered_names

    async def close(self) -> None:
        """Close all active MCP client connections."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
        self._tool_bindings.clear()

    def _build_executor(
        self,
        server_name: str,
        remote_tool_name: str,
    ) -> Any:
        async def execute_tool(
            ctx: dict[str, Any] | None = None,
            **arguments: Any,
        ) -> str:
            _ = ctx
            client = self._clients[server_name]
            result = await client.call_tool(remote_tool_name, arguments or None)
            return self._stringify_tool_result(result)

        return execute_tool

    def _stringify_tool_result(self, result: Any) -> str:
        """Render an MCP tool result for the chat transcript."""
        if isinstance(result, str):
            return result

        content = getattr(result, "content", None)
        if isinstance(content, list) and content:
            rendered_parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    rendered_parts.append(text)
                    continue

                try:
                    rendered = item.model_dump_json(indent=2)
                except AttributeError:
                    rendered = json.dumps(item, ensure_ascii=False, default=str)
                rendered_parts.append(str(rendered))
            return "\n\n".join(rendered_parts)

        structured_content = getattr(result, "structuredContent", None)
        if structured_content is not None:
            return json.dumps(structured_content, ensure_ascii=False, indent=2, default=str)

        try:
            rendered_result = result.model_dump_json(indent=2)
        except AttributeError:
            rendered_result = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return str(rendered_result)

    @staticmethod
    def _build_tool_name(server_name: str, remote_tool_name: str) -> str:
        """Create a namespaced local tool name for an MCP tool."""
        return f"mcp__{server_name}__{remote_tool_name}"
