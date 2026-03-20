"""Tests for workspace skill discovery."""

from pathlib import Path

from coding_agent.skills import SkillManager


def test_skill_manager_discovers_installed_workspace_skills(tmp_path: Path) -> None:
    """Workspace-local skills should be discovered from conventional roots."""
    skill_root = tmp_path / ".codex" / "skills" / "reviewer"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "# Reviewer\n\nReview code changes for regressions and missing tests.\n",
        encoding="utf-8",
    )

    manager = SkillManager(working_dir=tmp_path)
    catalog = manager.discover()

    assert [skill.name for skill in catalog.skills] == ["Reviewer"]
    assert catalog.prompt_entries(tmp_path)[0].path == ".codex\\skills\\reviewer\\SKILL.md"


def test_skill_manager_prefers_first_duplicate_skill_name(tmp_path: Path) -> None:
    """Earlier search roots should win when duplicate skill names are present."""
    first_skill = tmp_path / ".coding-agent" / "skills" / "deploy"
    second_skill = tmp_path / ".codex" / "skills" / "deploy"
    first_skill.mkdir(parents=True)
    second_skill.mkdir(parents=True)
    (first_skill / "SKILL.md").write_text(
        "# Deploy\n\nUse the workspace-local deployment workflow.\n",
        encoding="utf-8",
    )
    (second_skill / "SKILL.md").write_text(
        "# Deploy\n\nUse the fallback deployment workflow.\n",
        encoding="utf-8",
    )

    manager = SkillManager(working_dir=tmp_path)
    catalog = manager.discover()

    assert len(catalog.skills) == 1
    assert catalog.skills[0].description == "Use the workspace-local deployment workflow."


def test_skill_catalog_formats_markdown_summary(tmp_path: Path) -> None:
    """The skill catalog markdown summary should include roots and discovered skills."""
    skill_root = tmp_path / ".agents" / "skills" / "triage"
    skill_root.mkdir(parents=True)
    skill_file = skill_root / "SKILL.md"
    skill_file.write_text(
        "# Triage\n\nHandle issue triage for the current repository.\n",
        encoding="utf-8",
    )

    catalog = SkillManager(working_dir=tmp_path).discover()
    rendered = catalog.format_markdown(tmp_path)

    assert "# Skills" in rendered
    assert "Configured search roots: 3" in rendered
    assert "`Triage`" in rendered
    assert str(Path(".agents") / "skills" / "triage" / "SKILL.md") in rendered
