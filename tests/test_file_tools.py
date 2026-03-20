"""Tests for file tools."""

import tempfile

import pytest

from coding_agent.tools.file_tools import list_directory, read_file, write_file


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.mark.asyncio
async def test_read_write_file(temp_dir):
    """Test file read and write operations."""
    ctx = {"working_dir": temp_dir}

    # Write a file
    result = await write_file(
        path="test.txt",
        content="Hello, World!",
        ctx=ctx,
    )
    assert "Successfully wrote" in result

    # Read the file
    result = await read_file(
        path="test.txt",
        ctx=ctx,
    )
    assert "Hello, World!" in result


@pytest.mark.asyncio
async def test_read_file_offset_limit(temp_dir):
    """Test reading file with offset and limit."""
    ctx = {"working_dir": temp_dir}

    # Create a multi-line file
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    await write_file(path="lines.txt", content=content, ctx=ctx)

    # Read with offset and limit
    result = await read_file(path="lines.txt", offset=2, limit=2, ctx=ctx)
    assert "Line 2" in result
    assert "Line 3" in result
    assert "Line 1" not in result
    assert "Line 4" not in result


@pytest.mark.asyncio
async def test_list_directory(temp_dir):
    """Test directory listing."""
    ctx = {"working_dir": temp_dir}

    # Create some files
    await write_file(path="file1.txt", content="test", ctx=ctx)
    await write_file(path="file2.py", content="test", ctx=ctx)

    result = await list_directory(path=".", recursive=False, ctx=ctx)
    assert "file1.txt" in result
    assert "file2.py" in result


@pytest.mark.asyncio
async def test_path_safety(temp_dir):
    """Test path safety checks."""
    ctx = {"working_dir": temp_dir}

    # Try to access outside working directory
    result = await read_file(path="../outside.txt", ctx=ctx)
    assert "Access denied" in result
