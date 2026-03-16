"""Tests for tool registry."""

import pytest

from coding_agent.tools.registry import ToolRegistry, registry


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


def test_registry_list_tools():
    """Test listing tools."""
    tools = registry.list_tools()
    # Should have tools from imports
    assert len(tools) > 0
