"""Workspace-local skill discovery and rendering helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SkillSearchRoot:
    """A configured workspace-local directory searched for installed skills."""

    path: Path
    exists: bool


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    """Metadata about one discovered skill."""

    name: str
    description: str
    skill_file: Path
    root: Path


@dataclass(frozen=True, slots=True)
class SkillPromptEntry:
    """Prompt-friendly view of a discovered skill."""

    name: str
    description: str
    path: str


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    """Snapshot of the skills currently available in the workspace."""

    roots: tuple[SkillSearchRoot, ...] = ()
    skills: tuple[SkillDescriptor, ...] = ()

    def prompt_entries(self, working_dir: Path) -> tuple[SkillPromptEntry, ...]:
        """Return prompt-friendly entries with workspace-relative paths when possible."""
        return tuple(
            SkillPromptEntry(
                name=skill.name,
                description=skill.description,
                path=_display_path(skill.skill_file, working_dir),
            )
            for skill in self.skills
        )

    def root_display_paths(self, working_dir: Path) -> tuple[str, ...]:
        """Return display paths for configured skill search roots."""
        return tuple(_display_path(root.path, working_dir) for root in self.roots)

    def format_markdown(self, working_dir: Path) -> str:
        """Render a human-readable markdown summary of the discovered skills."""
        lines = [
            "# Skills",
            "",
            f"Configured search roots: {len(self.roots)}",
            f"Discovered skills: {len(self.skills)}",
        ]

        if self.roots:
            lines.extend(["", "Search roots:"])
            for root in self.roots:
                status = "present" if root.exists else "missing"
                lines.append(f"- `{_display_path(root.path, working_dir)}` ({status})")

        if self.skills:
            lines.extend(["", "Installed skills:"])
            for skill in self.skills:
                lines.append(
                    f"- `{skill.name}`: {skill.description} "
                    f"(`{_display_path(skill.skill_file, working_dir)}`)"
                )
        else:
            lines.extend(["", "No installed project skills were found."])

        return "\n".join(lines)


class SkillManager:
    """Discover installed skills under conventional workspace-local directories."""

    DEFAULT_SEARCH_ROOTS: tuple[str, ...] = (
        ".coding-agent/skills",
        ".codex/skills",
        ".agents/skills",
    )

    def __init__(
        self,
        *,
        working_dir: Path,
        search_roots: Sequence[Path] | None = None,
    ) -> None:
        self._working_dir = working_dir.resolve()
        self._search_roots = tuple(
            root.resolve() for root in (search_roots or self.default_search_roots(working_dir))
        )

    @classmethod
    def default_search_roots(cls, working_dir: Path) -> tuple[Path, ...]:
        """Return the workspace-local directories searched for installed skills."""
        workspace = working_dir.resolve()
        return tuple(
            (workspace / relative_root).resolve() for relative_root in cls.DEFAULT_SEARCH_ROOTS
        )

    def discover(self) -> SkillCatalog:
        """Scan the configured roots and return discovered skills."""
        roots = tuple(
            SkillSearchRoot(path=root, exists=root.exists() and root.is_dir())
            for root in self._search_roots
        )

        skills: list[SkillDescriptor] = []
        seen_names: set[str] = set()

        for root in roots:
            if not root.exists:
                continue

            for skill_file in _iter_skill_files(root.path):
                skill = _load_skill_descriptor(skill_file, root.path)
                if skill.name in seen_names:
                    continue
                seen_names.add(skill.name)
                skills.append(skill)

        return SkillCatalog(roots=roots, skills=tuple(skills))


def _iter_skill_files(root: Path) -> Iterable[Path]:
    """Yield installed skill manifests from one configured root."""
    direct_manifest = root / "SKILL.md"
    if direct_manifest.is_file():
        yield direct_manifest

    for skill_file in sorted(root.glob("*/SKILL.md")):
        if skill_file.is_file():
            yield skill_file


def _load_skill_descriptor(skill_file: Path, root: Path) -> SkillDescriptor:
    """Load display metadata from one SKILL.md manifest."""
    content = skill_file.read_text(encoding="utf-8")
    fallback_name = skill_file.parent.name
    return SkillDescriptor(
        name=_extract_skill_name(content, fallback_name),
        description=_extract_skill_description(content),
        skill_file=skill_file.resolve(),
        root=root.resolve(),
    )


def _extract_skill_name(content: str, fallback_name: str) -> str:
    """Extract the title for a skill from the first markdown heading."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped.removeprefix("# ").strip()
            if title:
                return title
            break
    return fallback_name


def _extract_skill_description(content: str) -> str:
    """Extract a short one-line summary from the manifest body."""
    in_code_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not stripped or stripped.startswith("#"):
            continue
        return stripped
    return "No description provided."


def _display_path(path: Path, working_dir: Path) -> str:
    """Render paths relative to the workspace when possible."""
    try:
        return str(path.resolve().relative_to(working_dir.resolve()))
    except ValueError:
        return str(path.resolve())
