"""File operation tools for the agent."""

from pathlib import Path
from typing import Any

import pathspec
from rich.console import Console

from coding_agent.config import settings
from coding_agent.tools.registry import registry

console = Console()


def _get_working_dir(ctx: dict[str, Any] | None) -> Path:
    """Get working directory from context or use current directory."""
    if ctx and "working_dir" in ctx:
        return Path(ctx["working_dir"]).resolve()
    return Path.cwd()


def _load_gitignore(working_dir: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns if exists."""
    gitignore_path = working_dir / ".gitignore"
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        return pathspec.PathSpec.from_lines("gitwildmatch", content.splitlines())
    return None


def _is_path_safe(path: Path, working_dir: Path) -> bool:
    """Check if path is within working directory."""
    try:
        path.resolve().relative_to(working_dir)
        return True
    except ValueError:
        return False


@registry.tool(
    name="read_file",
    description="Read the contents of a file. Returns the file content as a string. "
    "Use offset and limit to read specific portions of large files.",
)
async def read_file(
    path: str,
    offset: int = 1,
    limit: int | None = None,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Read file contents with optional line range."""
    working_dir = _get_working_dir(ctx)
    file_path = (working_dir / path).resolve()

    # Security check
    if not _is_path_safe(file_path, working_dir):
        return f"Error: Access denied. Path '{path}' is outside working directory."

    if not file_path.exists():
        return f"Error: File '{path}' not found."

    if not file_path.is_file():
        return f"Error: '{path}' is not a file."

    # Check file size
    file_size = file_path.stat().st_size
    if file_size > settings.max_file_size:
        return (
            f"Error: File '{path}' is too large "
            f"({file_size} bytes > {settings.max_file_size} bytes)."
        )

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Apply offset and limit (1-based indexing)
        start_idx = max(0, offset - 1)
        end_idx = len(lines) if limit is None else min(start_idx + limit, len(lines))

        selected_lines = lines[start_idx:end_idx]
        result = "\n".join(selected_lines)

        # Add truncation notice
        if start_idx > 0 or end_idx < len(lines):
            header = f"[Showing lines {start_idx + 1}-{end_idx} of {len(lines)}]\n```\n"
            footer = "\n```"
        else:
            header = "```\n"
            footer = "\n```"

        return f"{header}{result}{footer}"

    except UnicodeDecodeError:
        return f"Error: File '{path}' is not a text file."
    except Exception as e:
        return f"Error reading file: {e}"


@registry.tool(
    name="write_file",
    description="Write content to a file. Creates the file if it doesn't exist and "
    "overwrites any existing content. For large files, write the first chunk with "
    "this tool and append the remaining chunks with append_file.",
)
async def write_file(
    path: str,
    content: str,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Write content to a file."""
    working_dir = _get_working_dir(ctx)
    file_path = (working_dir / path).resolve()

    # Security check
    if not _is_path_safe(file_path, working_dir):
        return f"Error: Access denied. Path '{path}' is outside working directory."

    # Create parent directories if needed
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except Exception as e:
        return f"Error writing file: {e}"


@registry.tool(
    name="append_file",
    description="Append content to the end of a file. Creates the file if it doesn't "
    "exist. Use this after write_file when a large file needs to be written in "
    "multiple chunks.",
)
async def append_file(
    path: str,
    content: str,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Append content to a file."""
    working_dir = _get_working_dir(ctx)
    file_path = (working_dir / path).resolve()

    if not _is_path_safe(file_path, working_dir):
        return f"Error: Access denied. Path '{path}' is outside working directory."

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return f"Successfully appended {len(content)} characters to '{path}'."
    except Exception as e:
        return f"Error appending file: {e}"


@registry.tool(
    name="list_directory",
    description="List the contents of a directory. Returns files and subdirectories. "
    "Use recursive=true to list subdirectories as well.",
)
async def list_directory(
    path: str = ".",
    recursive: bool = False,
    ctx: dict[str, Any] | None = None,
) -> str:
    """List directory contents."""
    working_dir = _get_working_dir(ctx)
    dir_path = (working_dir / path).resolve()

    # Security check
    if not _is_path_safe(dir_path, working_dir):
        return f"Error: Access denied. Path '{path}' is outside working directory."

    if not dir_path.exists():
        return f"Error: Directory '{path}' not found."

    if not dir_path.is_dir():
        return f"Error: '{path}' is not a directory."

    gitignore = _load_gitignore(working_dir)

    try:
        entries = []

        if recursive:
            for item in sorted(dir_path.rglob("*")):
                rel_path = item.relative_to(working_dir)
                # Check gitignore
                if gitignore and gitignore.match_file(str(rel_path)):
                    continue

                depth = len(rel_path.parts) - 1
                indent = "  " * depth
                icon = "📁" if item.is_dir() else "📄"
                entries.append(f"{indent}{icon} {rel_path.name}")
        else:
            for item in sorted(dir_path.iterdir()):
                rel_path = item.relative_to(working_dir)
                # Check gitignore
                if gitignore and gitignore.match_file(str(rel_path)):
                    continue

                icon = "📁" if item.is_dir() else "📄"
                entries.append(f"{icon} {item.name}")

        if not entries:
            return f"Directory '{path}' is empty (or all files are ignored)."

        return f"Contents of '{path}':\n" + "\n".join(entries)

    except Exception as e:
        return f"Error listing directory: {e}"


@registry.tool(
    name="search_files",
    description="Search for files by name pattern (glob). Returns matching file paths. "
    "Example: '*.py' finds all Python files.",
)
async def search_files(
    pattern: str,
    path: str = ".",
    ctx: dict[str, Any] | None = None,
) -> str:
    """Search for files matching a glob pattern."""
    working_dir = _get_working_dir(ctx)
    search_path = (working_dir / path).resolve()

    # Security check
    if not _is_path_safe(search_path, working_dir):
        return f"Error: Access denied. Path '{path}' is outside working directory."

    try:
        matches = list(search_path.rglob(pattern))
        if not matches:
            return f"No files matching '{pattern}' found in '{path}'."

        # Sort and format results
        results = []
        for match in sorted(matches):
            rel_path = match.relative_to(working_dir)
            results.append(f"- {rel_path}")

        return f"Found {len(matches)} file(s) matching '{pattern}':\n" + "\n".join(results)

    except Exception as e:
        return f"Error searching files: {e}"


@registry.tool(
    name="grep_search",
    description="Search for text patterns within file contents (like grep). "
    "Returns matching file paths with line numbers and snippets.",
)
async def grep_search(
    pattern: str,
    path: str = ".",
    file_pattern: str = "*",
    ctx: dict[str, Any] | None = None,
) -> str:
    """Search for text pattern in file contents."""
    import re

    working_dir = _get_working_dir(ctx)
    search_path = (working_dir / path).resolve()

    # Security check
    if not _is_path_safe(search_path, working_dir):
        return f"Error: Access denied. Path '{path}' is outside working directory."

    try:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = []

        for file_path in search_path.rglob(file_pattern):
            if not file_path.is_file():
                continue

            # Skip binary/large files
            if file_path.stat().st_size > settings.max_file_size:
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.splitlines()

                file_matches = []
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        # Show context: 2 lines before and after
                        start = max(0, i - 3)
                        end = min(len(lines), i + 2)
                        context = lines[start:end]
                        context_str = "\n".join(
                            f"{start + j + 1}: {ctx_line}" for j, ctx_line in enumerate(context)
                        )
                        file_matches.append(f"  Line {i}:\n{context_str}")

                if file_matches:
                    rel_path = file_path.relative_to(working_dir)
                    matches.append(f"📄 {rel_path}\n" + "\n".join(file_matches))

            except UnicodeDecodeError, Exception:
                continue

        if not matches:
            return f"No matches found for pattern '{pattern}'."

        return f"Found matches for '{pattern}':\n\n" + "\n\n".join(matches[:10])  # Limit results

    except re.error as e:
        return f"Error: Invalid regex pattern - {e}"
    except Exception as e:
        return f"Error searching: {e}"
