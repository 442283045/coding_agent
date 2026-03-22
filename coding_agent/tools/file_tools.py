"""File operation tools for the agent."""

from pathlib import Path
from typing import Any

import pathspec
from pydantic import BaseModel, ConfigDict, Field, model_validator

from coding_agent.config import settings
from coding_agent.tools.registry import registry


class _FileToolInput(BaseModel):
    """Shared validation rules for workspace-bounded file tools."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        description="Relative path to a file inside the working directory.",
    )


class ReadFileInput(_FileToolInput):
    """Validated arguments for reading a text file."""

    offset: int = Field(
        default=1,
        ge=1,
        description="1-based starting line number to read from.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of lines to read. Omit to read through the end.",
    )


class WriteFileInput(_FileToolInput):
    """Validated arguments for creating or fully rewriting a file."""

    content: str = Field(description="Complete text content to write to the file.")
    overwrite: bool = Field(
        default=False,
        description=(
            "Whether to replace an existing file. Leave false for new files and use "
            "patch_file or replace_text for focused edits."
        ),
    )


class AppendFileInput(_FileToolInput):
    """Validated arguments for appending to an existing or new file."""

    content: str = Field(description="Text content to append to the end of the file.")
    create: bool = Field(
        default=True,
        description="Whether to create the file when it does not already exist.",
    )


class PatchFileInput(_FileToolInput):
    """Validated arguments for line-range replacements."""

    start_line: int = Field(
        ge=1,
        description="1-based starting line number of the inclusive range to replace.",
    )
    end_line: int = Field(
        ge=1,
        description="1-based ending line number of the inclusive range to replace.",
    )
    new_content: str = Field(
        description=(
            "Replacement text for the selected line range. Use an empty string to delete it."
        ),
    )
    expected_old_text: str | None = Field(
        default=None,
        description=(
            "Optional exact text currently expected in the selected line range. "
            "Use the text returned by read_file to detect drift before patching."
        ),
    )

    @model_validator(mode="after")
    def validate_line_range(self) -> PatchFileInput:
        """Ensure the requested patch range is well formed."""
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line.")
        return self


class ReplaceTextInput(_FileToolInput):
    """Validated arguments for exact text replacement."""

    old_text: str = Field(
        min_length=1,
        description="Exact existing text to replace. Read it from the file first to avoid drift.",
    )
    new_text: str = Field(
        description="Replacement text. Use an empty string to delete the matched text.",
    )
    occurrence: int | None = Field(
        default=None,
        ge=1,
        description=(
            "1-based occurrence to replace when the old text appears multiple times. "
            "Omit to require the match count specified by expected_replacements."
        ),
    )
    expected_replacements: int = Field(
        default=1,
        ge=1,
        description=(
            "Expected number of matches when occurrence is omitted. Defaults to 1 to "
            "prevent ambiguous replacements."
        ),
    )


def _resolve_workspace_path(path: str, working_dir: Path) -> Path:
    """Resolve a relative workspace path against the active working directory."""
    return (working_dir / path).resolve()


def _get_working_dir(ctx: dict[str, Any] | None) -> Path:
    """Get working directory from context or use current directory."""
    if ctx and "working_dir" in ctx:
        return Path(ctx["working_dir"]).resolve()
    return Path.cwd()


def _load_gitignore(working_dir: Path) -> pathspec.GitIgnoreSpec | None:
    """Load .gitignore patterns when the workspace defines them."""
    gitignore_path = working_dir / ".gitignore"
    if not gitignore_path.is_file():
        return None

    content = gitignore_path.read_text(encoding="utf-8")
    return pathspec.GitIgnoreSpec.from_lines(content.splitlines())


def _is_gitignored(
    rel_path: Path,
    *,
    gitignore: pathspec.GitIgnoreSpec | None,
    is_dir: bool,
) -> bool:
    """Check whether a workspace-relative path should be filtered by .gitignore."""
    if gitignore is None:
        return False

    normalized_path = rel_path.as_posix()
    if is_dir and not normalized_path.endswith("/"):
        normalized_path += "/"

    return bool(gitignore.match_file(normalized_path))


def _is_path_safe(path: Path, working_dir: Path) -> bool:
    """Check if path is within working directory."""
    try:
        path.resolve().relative_to(working_dir)
        return True
    except ValueError:
        return False


def _normalize_newlines(text: str) -> str:
    """Normalize all newlines in a text buffer to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_match_text(text: str) -> str:
    """Normalize text for exact-match comparisons across newline styles."""
    return _normalize_newlines(text).rstrip("\n")


def _detect_newline(content: str) -> str:
    """Detect the dominant newline style for a text buffer."""
    if "\r\n" in content:
        return "\r\n"
    if "\r" in content:
        return "\r"
    return "\n"


def _normalize_patch_content(
    new_content: str,
    *,
    newline: str,
    has_suffix: bool,
) -> str:
    """Normalize replacement content while preserving line boundaries."""
    if not new_content:
        return ""

    normalized = new_content.replace("\r\n", "\n").replace("\r", "\n")
    replacement = normalized.replace("\n", newline)

    if has_suffix and not replacement.endswith(newline):
        replacement += newline

    return replacement


def _truncate_preview(text: str, *, limit: int = 400) -> str:
    """Trim a block preview so error messages stay readable."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _replace_nth_occurrence(content: str, old_text: str, new_text: str, occurrence: int) -> str:
    """Replace the requested 1-based occurrence of a substring."""
    search_start = 0
    match_start = -1

    for _ in range(occurrence):
        match_start = content.find(old_text, search_start)
        if match_start == -1:
            raise ValueError("occurrence is outside the number of matches in the file.")
        search_start = match_start + len(old_text)

    return (
        f"{content[:match_start]}{new_text}{content[match_start + len(old_text) :]}"
        if match_start >= 0
        else content
    )


@registry.tool(
    name="read_file",
    description="Read the contents of a file. Returns the file content as a string. "
    "Use offset and limit to read specific portions of large files.",
    input_model=ReadFileInput,
)
async def read_file(
    path: str,
    offset: int = 1,
    limit: int | None = None,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Read file contents with optional line range."""
    working_dir = _get_working_dir(ctx)
    file_path = _resolve_workspace_path(path, working_dir)

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
    description="Create a new text file or fully rewrite an existing one. By default, "
    "this refuses to overwrite existing files; use overwrite=true only after reviewing "
    "the current file. For local edits, prefer replace_text or patch_file.",
    input_model=WriteFileInput,
)
async def write_file(
    path: str,
    content: str,
    overwrite: bool = False,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Write content to a file."""
    working_dir = _get_working_dir(ctx)
    file_path = _resolve_workspace_path(path, working_dir)

    # Security check
    if not _is_path_safe(file_path, working_dir):
        return f"Error: Access denied. Path '{path}' is outside working directory."

    if file_path.exists() and not file_path.is_file():
        return f"Error: '{path}' is not a file."

    if file_path.exists() and not overwrite:
        return (
            f"Error: File '{path}' already exists. Use replace_text or patch_file for "
            "focused edits, or retry write_file with overwrite=true after reviewing it."
        )

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
    "exist by default. Use this after write_file when a large file needs to be written "
    "in multiple chunks.",
    input_model=AppendFileInput,
)
async def append_file(
    path: str,
    content: str,
    create: bool = True,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Append content to a file."""
    working_dir = _get_working_dir(ctx)
    file_path = _resolve_workspace_path(path, working_dir)

    if not _is_path_safe(file_path, working_dir):
        return f"Error: Access denied. Path '{path}' is outside working directory."

    if file_path.exists() and not file_path.is_file():
        return f"Error: '{path}' is not a file."

    if not file_path.exists() and not create:
        return f"Error: File '{path}' not found. Retry with create=true to create it first."

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return f"Successfully appended {len(content)} characters to '{path}'."
    except Exception as e:
        return f"Error appending file: {e}"


@registry.tool(
    name="patch_file",
    description="Replace a contiguous inclusive line range in an existing text file "
    "without rewriting the whole file. Read the relevant lines first with read_file, "
    "then patch only that region. Provide expected_old_text from read_file to detect "
    "drift before writing.",
    input_model=PatchFileInput,
)
async def patch_file(
    path: str,
    start_line: int,
    end_line: int,
    new_content: str,
    expected_old_text: str | None = None,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Replace a line range in a text file."""
    working_dir = _get_working_dir(ctx)
    file_path = _resolve_workspace_path(path, working_dir)

    if not _is_path_safe(file_path, working_dir):
        return f"Error: Access denied. Path '{path}' is outside working directory."

    if not file_path.exists():
        return f"Error: File '{path}' not found."

    if not file_path.is_file():
        return f"Error: '{path}' is not a file."

    if start_line < 1:
        return "Error: start_line must be at least 1."

    if end_line < start_line:
        return "Error: end_line must be greater than or equal to start_line."

    file_size = file_path.stat().st_size
    if file_size > settings.max_file_size:
        return (
            f"Error: File '{path}' is too large "
            f"({file_size} bytes > {settings.max_file_size} bytes)."
        )

    try:
        content = file_path.read_text(encoding="utf-8", newline="")
        lines = content.splitlines(keepends=True)

        if not lines:
            return f"Error: File '{path}' is empty. Use write_file to create new content."

        if end_line > len(lines):
            return (
                f"Error: Line range {start_line}-{end_line} is outside "
                f"'{path}' ({len(lines)} lines)."
            )

        selected_text = "".join(lines[start_line - 1 : end_line])
        if expected_old_text is not None and _normalize_match_text(
            selected_text
        ) != _normalize_match_text(expected_old_text):
            current_preview = _truncate_preview(_normalize_newlines(selected_text))
            return (
                "Error: Patch conflict. The requested line range no longer matches "
                "expected_old_text. Re-read the file before retrying.\nCurrent text:\n```\n"
                f"{current_preview}\n```"
            )

        newline = _detect_newline(content)
        prefix = "".join(lines[: start_line - 1])
        suffix = "".join(lines[end_line:])
        replacement = _normalize_patch_content(
            new_content,
            newline=newline,
            has_suffix=bool(suffix),
        )
        updated_content = f"{prefix}{replacement}{suffix}"

        file_path.write_text(updated_content, encoding="utf-8", newline="")

        replacement_lines = 0 if not replacement else len(replacement.splitlines())
        replaced_line_count = end_line - start_line + 1
        return (
            f"Successfully patched '{path}' replacing lines {start_line}-{end_line} "
            f"({replaced_line_count} line(s)) with {replacement_lines} line(s)."
        )
    except UnicodeDecodeError:
        return f"Error: File '{path}' is not a text file."
    except Exception as e:
        return f"Error patching file: {e}"


@registry.tool(
    name="replace_text",
    description="Replace an exact text block in an existing file. Read the target text "
    "first, then pass that exact old_text. This is safer than full-file rewrites and "
    "avoids brittle line numbers for local edits.",
    input_model=ReplaceTextInput,
)
async def replace_text(
    path: str,
    old_text: str,
    new_text: str,
    occurrence: int | None = None,
    expected_replacements: int = 1,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Replace exact text within an existing text file."""
    working_dir = _get_working_dir(ctx)
    file_path = _resolve_workspace_path(path, working_dir)

    if not _is_path_safe(file_path, working_dir):
        return f"Error: Access denied. Path '{path}' is outside working directory."

    if not file_path.exists():
        return f"Error: File '{path}' not found."

    if not file_path.is_file():
        return f"Error: '{path}' is not a file."

    file_size = file_path.stat().st_size
    if file_size > settings.max_file_size:
        return (
            f"Error: File '{path}' is too large "
            f"({file_size} bytes > {settings.max_file_size} bytes)."
        )

    try:
        content = file_path.read_text(encoding="utf-8", newline="")
        newline = _detect_newline(content)
        normalized_content = _normalize_newlines(content)
        normalized_old_text = _normalize_match_text(old_text)
        normalized_new_text = _normalize_newlines(new_text)

        match_count = normalized_content.count(normalized_old_text)
        if match_count == 0:
            return (
                f"Error: old_text was not found in '{path}'. Re-read the file and copy "
                "the exact text you want to replace."
            )

        if occurrence is not None:
            if occurrence > match_count:
                return (
                    f"Error: occurrence {occurrence} is outside the {match_count} match(es) "
                    f"found in '{path}'."
                )
            updated_content = _replace_nth_occurrence(
                normalized_content,
                normalized_old_text,
                normalized_new_text,
                occurrence,
            )
            replaced_count = 1
        else:
            if match_count != expected_replacements:
                return (
                    f"Error: replace_text found {match_count} match(es) in '{path}', but "
                    f"expected {expected_replacements}. Narrow old_text or set occurrence."
                )
            updated_content = normalized_content.replace(normalized_old_text, normalized_new_text)
            replaced_count = match_count

        file_path.write_text(updated_content.replace("\n", newline), encoding="utf-8", newline="")
        return f"Successfully replaced {replaced_count} occurrence(s) in '{path}'."
    except UnicodeDecodeError:
        return f"Error: File '{path}' is not a text file."
    except Exception as e:
        return f"Error replacing text: {e}"


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
                if _is_gitignored(rel_path, gitignore=gitignore, is_dir=item.is_dir()):
                    continue

                depth = len(rel_path.parts) - 1
                indent = "  " * depth
                icon = "📁" if item.is_dir() else "📄"
                entries.append(f"{indent}{icon} {rel_path.name}")
        else:
            for item in sorted(dir_path.iterdir()):
                rel_path = item.relative_to(working_dir)
                if _is_gitignored(rel_path, gitignore=gitignore, is_dir=item.is_dir()):
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
