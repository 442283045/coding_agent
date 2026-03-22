"""Tests for file tools."""

import tempfile
import warnings
from pathlib import Path

import pytest

from coding_agent.tools.file_tools import (
    append_file,
    list_directory,
    patch_file,
    read_file,
    replace_text,
    write_file,
)
from coding_agent.tools.registry import registry


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
async def test_list_directory_respects_gitignore_without_deprecation_warning(temp_dir):
    """Directory listing should honor .gitignore without pathspec deprecation warnings."""
    ctx = {"working_dir": temp_dir}
    root = Path(temp_dir)

    (root / ".gitignore").write_text("ignored.txt\nignored_dir/\n", encoding="utf-8")
    await write_file(path="visible.txt", content="keep me", ctx=ctx)
    await write_file(path="ignored.txt", content="hide me", ctx=ctx)
    ignored_dir = root / "ignored_dir"
    ignored_dir.mkdir()
    (ignored_dir / "nested.txt").write_text("nested", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await list_directory(path=".", recursive=True, ctx=ctx)

    assert "visible.txt" in result
    assert "ignored.txt" not in result
    assert "ignored_dir" not in result
    assert not any(
        issubclass(warning.category, DeprecationWarning) and "gitwildmatch" in str(warning.message)
        for warning in caught
    )


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
async def test_write_file_refuses_to_overwrite_existing_file_by_default(temp_dir):
    """write_file should protect existing files unless overwrite is explicit."""
    ctx = {"working_dir": temp_dir}

    await write_file(path="safe.txt", content="original", ctx=ctx)
    result = await write_file(path="safe.txt", content="replacement", ctx=ctx)

    assert "already exists" in result
    assert Path(temp_dir, "safe.txt").read_text(encoding="utf-8") == "original"


@pytest.mark.asyncio
async def test_write_file_can_overwrite_when_explicitly_enabled(temp_dir):
    """write_file should still support full rewrites when overwrite=true."""
    ctx = {"working_dir": temp_dir}

    await write_file(path="rewrite.txt", content="old", ctx=ctx)
    result = await write_file(
        path="rewrite.txt",
        content="new",
        overwrite=True,
        ctx=ctx,
    )

    assert "Successfully wrote" in result
    assert Path(temp_dir, "rewrite.txt").read_text(encoding="utf-8") == "new"


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
async def test_patch_file_detects_expected_old_text_conflicts(temp_dir):
    """expected_old_text should prevent stale line-range patches from landing."""
    ctx = {"working_dir": temp_dir}
    content = "first\nsecond\nthird\n"
    await write_file(path="conflict.txt", content=content, ctx=ctx)

    result = await patch_file(
        path="conflict.txt",
        start_line=2,
        end_line=2,
        new_content="updated",
        expected_old_text="stale text",
        ctx=ctx,
    )

    assert "Patch conflict" in result
    assert Path(temp_dir, "conflict.txt").read_text(encoding="utf-8") == content


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


@pytest.mark.asyncio
async def test_replace_text_replaces_a_unique_match(temp_dir):
    """replace_text should safely replace an exact block without line numbers."""
    ctx = {"working_dir": temp_dir}
    await write_file(path="replace.py", content="alpha\nbeta\ngamma\n", ctx=ctx)

    result = await replace_text(
        path="replace.py",
        old_text="beta",
        new_text="delta",
        ctx=ctx,
    )

    assert "Successfully replaced 1 occurrence" in result
    assert Path(temp_dir, "replace.py").read_text(encoding="utf-8") == "alpha\ndelta\ngamma\n"


@pytest.mark.asyncio
async def test_replace_text_rejects_ambiguous_matches_without_occurrence(temp_dir):
    """replace_text should reject ambiguous blocks unless the caller narrows the match."""
    ctx = {"working_dir": temp_dir}
    await write_file(path="ambiguous.txt", content="x\nrepeat\nrepeat\n", ctx=ctx)

    result = await replace_text(
        path="ambiguous.txt",
        old_text="repeat",
        new_text="done",
        ctx=ctx,
    )

    assert "expected 1" in result


@pytest.mark.asyncio
async def test_replace_text_can_target_a_specific_occurrence(temp_dir):
    """replace_text should handle repeated blocks when occurrence is specified."""
    ctx = {"working_dir": temp_dir}
    await write_file(path="occurrence.txt", content="repeat\nrepeat\n", ctx=ctx)

    result = await replace_text(
        path="occurrence.txt",
        old_text="repeat",
        new_text="done",
        occurrence=2,
        ctx=ctx,
    )

    assert "Successfully replaced 1 occurrence" in result
    assert Path(temp_dir, "occurrence.txt").read_text(encoding="utf-8") == "repeat\ndone\n"


@pytest.mark.asyncio
async def test_read_file_tool_validation_rejects_non_integer_limit(temp_dir):
    """Tool execution should validate read_file arguments before the function runs."""
    ctx = {"working_dir": temp_dir}
    await write_file(path="typed.txt", content="a\nb\n", ctx=ctx)
    tool = registry.get("read_file")

    assert tool is not None

    with pytest.raises(ValueError):
        await tool.execute(path="typed.txt", offset=1, limit="all", ctx=ctx)
