"""Tests for tool registry."""

import pytest
from pydantic import BaseModel, ConfigDict, Field

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


@pytest.mark.asyncio
async def test_tool_input_model_validates_and_exports_schema(clean_registry) -> None:
    """Decorator-based tools should validate inputs with a Pydantic model."""

    class SearchInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        query: str = Field(description="Search query")
        limit: int = Field(default=10, ge=1, description="Result limit")

    @clean_registry.tool("search", "Search tool", input_model=SearchInput)
    async def search(query: str, limit: int = 10) -> str:
        return f"{query}:{limit}"

    tool = clean_registry.get("search")

    assert tool is not None
    assert await tool.execute(query="docs", limit="3") == "docs:3"
    assert tool.to_openai_format()["function"]["parameters"]["properties"]["limit"]["minimum"] == 1

    with pytest.raises(ValueError):
        await tool.execute(query="docs", limit="zero")
