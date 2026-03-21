"""Helpers for building user-visible previews of file changes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Literal

FILE_EDIT_TOOL_NAMES = frozenset({"write_file", "append_file", "patch_file"})
_DIFF_PREVIEW_LINE_LIMIT = 240
_DIFF_PREVIEW_CHAR_LIMIT = 12_000


@dataclass(slots=True)
class FileSnapshot:
    """A file state captured before a mutating tool runs."""

    path: Path
    display_path: str
    existed: bool
    content: str


@dataclass(slots=True)
class FileChangePreview:
    """A rendered summary of a file change for the CLI."""

    path: str
    operation: Literal["created", "updated"]
    diff: str
    added_lines: int
    removed_lines: int
    truncated: bool = False


def _is_path_within_working_dir(path: Path, working_dir: Path) -> bool:
    """Return whether a resolved path stays within the working directory."""
    try:
        path.relative_to(working_dir)
    except ValueError:
        return False
    return True


def _display_path(path: Path, working_dir: Path) -> str:
    """Return a stable relative path for UI display."""
    try:
        return path.relative_to(working_dir).as_posix()
    except ValueError:
        return path.name


def capture_file_snapshot(
    tool_name: str,
    *,
    args: Mapping[str, object],
    working_dir: Path,
) -> FileSnapshot | None:
    """Capture the pre-edit state for supported file mutation tools."""
    if tool_name not in FILE_EDIT_TOOL_NAMES:
        return None

    path_value = args.get("path")
    if not isinstance(path_value, str) or not path_value:
        return None

    file_path = (working_dir / path_value).resolve()
    if not _is_path_within_working_dir(file_path, working_dir):
        return None

    display_path = _display_path(file_path, working_dir)
    if not file_path.exists():
        return FileSnapshot(path=file_path, display_path=display_path, existed=False, content="")

    if not file_path.is_file():
        return None

    try:
        content = file_path.read_text(encoding="utf-8", newline="")
    except UnicodeDecodeError:
        return None

    return FileSnapshot(
        path=file_path,
        display_path=display_path,
        existed=True,
        content=content,
    )


def _truncate_diff_lines(diff_lines: Sequence[str]) -> tuple[str, bool]:
    """Trim oversized diff previews for terminal rendering."""
    visible_lines: list[str] = []
    char_count = 0

    for index, line in enumerate(diff_lines):
        next_char_count = char_count + len(line) + (1 if visible_lines else 0)
        if index >= _DIFF_PREVIEW_LINE_LIMIT or next_char_count > _DIFF_PREVIEW_CHAR_LIMIT:
            visible_lines.append("... diff truncated ...")
            return "\n".join(visible_lines), True
        visible_lines.append(line)
        char_count = next_char_count

    return "\n".join(visible_lines), False


def build_file_change_preview(
    snapshot: FileSnapshot | None,
    *,
    result: str,
) -> FileChangePreview | None:
    """Build a unified diff preview after a successful file edit."""
    if snapshot is None or result.startswith("Error"):
        return None

    if not snapshot.path.exists() or not snapshot.path.is_file():
        return None

    try:
        updated_content = snapshot.path.read_text(encoding="utf-8", newline="")
    except UnicodeDecodeError:
        return None

    if updated_content == snapshot.content:
        return None

    before_lines = snapshot.content.splitlines()
    after_lines = updated_content.splitlines()
    diff_lines = list(
        unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{snapshot.display_path}" if snapshot.existed else "/dev/null",
            tofile=f"b/{snapshot.display_path}",
            n=3,
            lineterm="",
        )
    )
    if not diff_lines:
        return None

    diff_text, truncated = _truncate_diff_lines(diff_lines)
    added_lines = sum(
        1 for line in diff_lines if line.startswith("+") and not line.startswith("+++")
    )
    removed_lines = sum(
        1 for line in diff_lines if line.startswith("-") and not line.startswith("---")
    )
    operation: Literal["created", "updated"] = "updated" if snapshot.existed else "created"

    return FileChangePreview(
        path=snapshot.display_path,
        operation=operation,
        diff=diff_text,
        added_lines=added_lines,
        removed_lines=removed_lines,
        truncated=truncated,
    )
