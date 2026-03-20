"""Tests for MCP client integration."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from coding_agent.config import MCPServerConfig
from coding_agent.mcp.manager import MCPManager
from coding_agent.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_mcp_manager_registers_and_executes_dynamic_tools(monkeypatch) -> None:
    """Configured MCP tools should be added to the runtime registry and stay callable."""

    recorded_calls: list[tuple[str, dict[str, object] | None]] = []

    class FakeClient:
        def __init__(self, transport: dict[str, object]) -> None:
            self.transport = transport

        async def __aenter__(self) -> FakeClient:
            return self

        async def list_tools(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    name="search",
                    title="Search",
                    description="Search documents",
                    inputSchema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                )
            ]

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object] | None = None,
        ) -> SimpleNamespace:
            recorded_calls.append((name, arguments))
            return SimpleNamespace(
                content=[SimpleNamespace(text="match one"), SimpleNamespace(text="match two")]
            )

        async def close(self) -> None:
            return None

    fake_fastmcp = ModuleType("fastmcp")
    fake_fastmcp.Client = FakeClient
    monkeypatch.setitem(sys.modules, "fastmcp", fake_fastmcp)

    registry = ToolRegistry()
    manager = MCPManager(
        {
            "docs": MCPServerConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "."],
            )
        }
    )

    registered_names = await manager.start(registry)
    tool = registry.get("mcp__docs__search")

    assert registered_names == ["mcp__docs__search"]
    assert tool is not None
    assert tool.to_openai_format()["function"]["parameters"]["required"] == ["query"]

    result = await tool.execute(query="agents.md")

    assert result == "match one\n\nmatch two"
    assert recorded_calls == [("search", {"query": "agents.md"})]

    await manager.close()
