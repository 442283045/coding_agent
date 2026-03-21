"""Tests for terminal diff previews of file edits."""

from pathlib import Path

from coding_agent.utils.file_diff import build_file_change_preview, capture_file_snapshot


def test_build_file_change_preview_for_created_file(tmp_path: Path) -> None:
    """A newly created file should produce an all-additions diff preview."""
    snapshot = capture_file_snapshot("write_file", args={"path": "demo.py"}, working_dir=tmp_path)

    assert snapshot is not None
    assert snapshot.existed is False

    (tmp_path / "demo.py").write_text("print('hi')\n", encoding="utf-8")
    preview = build_file_change_preview(
        snapshot,
        result="Successfully wrote 12 characters to 'demo.py'.",
    )

    assert preview is not None
    assert preview.operation == "created"
    assert preview.path == "demo.py"
    assert preview.added_lines == 1
    assert preview.removed_lines == 0
    assert "+++ b/demo.py" in preview.diff
    assert "+print('hi')" in preview.diff


def test_build_file_change_preview_truncates_large_diffs(tmp_path: Path) -> None:
    """Oversized previews should be truncated to keep the CLI responsive."""
    snapshot = capture_file_snapshot("write_file", args={"path": "big.txt"}, working_dir=tmp_path)

    assert snapshot is not None

    content = "\n".join(f"line {index}" for index in range(400))
    (tmp_path / "big.txt").write_text(content, encoding="utf-8")
    preview = build_file_change_preview(
        snapshot,
        result="Successfully wrote 3489 characters to 'big.txt'.",
    )

    assert preview is not None
    assert preview.truncated is True
    assert preview.diff.endswith("... diff truncated ...")
