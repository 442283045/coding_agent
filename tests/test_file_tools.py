"""Tests for file tools."""

import tempfile
from pathlib import Path

import pytest

from coding_agent.tools.file_tools import (
    append_file,
    list_directory,
    patch_file,
    read_file,
    write_file,
)


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


@pytest.mark.asyncio
async def test_append_file(temp_dir):
    """Appending should preserve existing content and create the file if needed."""
    ctx = {"working_dir": temp_dir}

    await write_file(path="story.txt", content="Hello", ctx=ctx)
    result = await append_file(path="story.txt", content=", world!", ctx=ctx)

    assert "Successfully appended" in result

    content = await read_file(path="story.txt", ctx=ctx)
    assert "Hello, world!" in content


@pytest.mark.asyncio
async def test_patch_file_replaces_selected_lines(temp_dir):
    """Patching should replace only the requested line range."""
    ctx = {"working_dir": temp_dir}
    content = "\n".join(
        [
            "import os",
            "",
            "def old_helper():",
            "    return 'old'",
            "",
            "print(old_helper())",
            "",
        ]
    )
    await write_file(path="module.py", content=content, ctx=ctx)

    result = await patch_file(
        path="module.py",
        start_line=3,
        end_line=4,
        new_content="def new_helper():\n    return 'new'",
        ctx=ctx,
    )

    assert "Successfully patched" in result

    updated = Path(temp_dir, "module.py").read_text(encoding="utf-8")
    assert "def old_helper():" not in updated
    assert "return 'old'" not in updated
    assert "def new_helper():" in updated
    assert "print(old_helper())" in updated


@pytest.mark.asyncio
async def test_patch_file_can_delete_a_module_block(temp_dir):
    """Patching with an empty replacement should delete the selected lines."""
    ctx = {"working_dir": temp_dir}
    content = "\n".join(
        [
            "def keep():",
            "    return 'keep'",
            "",
            "def remove_me():",
            "    return 'remove'",
            "",
            "print(keep())",
            "",
        ]
    )
    await write_file(path="delete_block.py", content=content, ctx=ctx)

    result = await patch_file(
        path="delete_block.py",
        start_line=4,
        end_line=5,
        new_content="",
        ctx=ctx,
    )

    assert "Successfully patched" in result

    updated = Path(temp_dir, "delete_block.py").read_text(encoding="utf-8")
    assert "def remove_me():" not in updated
    assert "return 'remove'" not in updated
    assert "def keep():" in updated
    assert "print(keep())" in updated


@pytest.mark.asyncio
async def test_patch_file_preserves_existing_line_endings(temp_dir):
    """Patching should preserve CRLF files instead of rewriting them with LF."""
    ctx = {"working_dir": temp_dir}
    file_path = Path(temp_dir, "windows.py")
    file_path.write_text("line1\r\nline2\r\nline3\r\n", encoding="utf-8", newline="")

    result = await patch_file(
        path="windows.py",
        start_line=2,
        end_line=2,
        new_content="updated_line",
        ctx=ctx,
    )

    assert "Successfully patched" in result
    assert file_path.read_text(encoding="utf-8", newline="") == "line1\r\nupdated_line\r\nline3\r\n"


@pytest.mark.asyncio
async def test_patch_file_rejects_invalid_ranges(temp_dir):
    """Invalid patch ranges should return actionable errors."""
    ctx = {"working_dir": temp_dir}
    await write_file(path="range.py", content="a\nb\n", ctx=ctx)

    result = await patch_file(
        path="range.py",
        start_line=3,
        end_line=2,
        new_content="",
        ctx=ctx,
    )
    assert "end_line must be greater than or equal to start_line" in result

    out_of_range = await patch_file(
        path="range.py",
        start_line=2,
        end_line=5,
        new_content="x",
        ctx=ctx,
    )
    assert "outside 'range.py'" in out_of_range
