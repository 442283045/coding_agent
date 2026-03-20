"""Tests for tool registry."""

import pytest

from coding_agent.tools.registry import Tool, ToolRegistry, registry


@pytest.fixture
def clean_registry():
    """Create a fresh registry for testing."""
    return ToolRegistry()


@pytest.mark.asyncio
async def test_tool_registration(clean_registry):
    """Test tool registration."""

    @clean_registry.tool("test_tool", "A test tool")
    async def test_tool(name: str) -> str:
        return f"Hello, {name}!"

    tool = clean_registry.get("test_tool")
    assert tool is not None
    assert tool.name == "test_tool"

    result = await tool.execute(name="World")
    assert result == "Hello, World!"


@pytest.mark.asyncio
async def test_tool_optional_params(clean_registry):
    """Test tool with optional parameters."""

    @clean_registry.tool("greet", "Greeting tool")
    async def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    tool = clean_registry.get("greet")
    result = await tool.execute(name="Alice")
    assert result == "Hello, Alice!"

    result = await tool.execute(name="Bob", greeting="Hi")
    assert result == "Hi, Bob!"


@pytest.mark.asyncio
async def test_registry_copy_keeps_detached_tool_definitions(clean_registry) -> None:
    """Registry copies should isolate later dynamic tool changes."""

    @clean_registry.tool("local_tool", "Local tool")
    async def local_tool(name: str) -> str:
        return name

    copied_registry = clean_registry.copy()
    clean_registry.unregister("local_tool")

    copied_tool = copied_registry.get("local_tool")

    assert copied_tool is not None
    assert await copied_tool.execute(name="Alice") == "Alice"


def test_tool_prefers_explicit_input_schema() -> None:
    """Dynamic tools should preserve their full JSON schema."""

    async def execute_tool(**kwargs: object) -> str:
        return str(kwargs)

    tool = Tool(
        name="mcp__demo__search",
        description="Search through MCP",
        func=execute_tool,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
        },
        source="mcp",
    )

    openai_tool = tool.to_openai_format()
    anthropic_tool = tool.to_anthropic_format()

    assert openai_tool["function"]["parameters"]["properties"]["limit"]["minimum"] == 1
    assert anthropic_tool["input_schema"]["required"] == ["query"]


def test_registry_list_tools():
    """Test listing tools."""
    from coding_agent.tools import code_tools, file_tools, shell_tools

    _ = (code_tools, file_tools, shell_tools)
    tools = registry.list_tools()
    assert len(tools) > 0
