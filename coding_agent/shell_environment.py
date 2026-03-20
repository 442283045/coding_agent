"""Shell environment detection shared by prompts and shell tools."""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ShellProfile:
    """Description of the shell used for command execution."""

    operating_system: str
    shell_name: str
    executable: str
    arguments: tuple[str, ...]
    command_style: str
    environment_syntax: str


def detect_shell_profile() -> ShellProfile:
    """Detect the shell profile the agent should use for command execution."""
    system_name = platform.system()

    if system_name == "Windows":
        executable = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        return ShellProfile(
            operating_system="Windows",
            shell_name="PowerShell",
            executable=executable,
            arguments=("-NoLogo", "-NoProfile", "-Command"),
            command_style=(
                "Use PowerShell syntax, not bash. Prefer cmdlets such as "
                "`Get-ChildItem`, `Get-Content`, and `Select-String` when appropriate."
            ),
            environment_syntax="Set environment variables with `$env:NAME='value'`.",
        )

    executable = shutil.which("bash") or "/bin/bash"
    operating_system = "macOS" if system_name == "Darwin" else "Linux"
    return ShellProfile(
        operating_system=operating_system,
        shell_name="bash",
        executable=executable,
        arguments=("-lc",),
        command_style=(
            "Use bash syntax, not PowerShell. Prefer standard POSIX utilities such as "
            "`ls`, `cat`, `grep`, and `find` when appropriate."
        ),
        environment_syntax="Set environment variables with `export NAME=value`.",
    )


def resolve_shell_executable(profile: ShellProfile) -> str:
    """Return the executable path or command name used to launch the shell."""
    shell_path = Path(profile.executable)
    if shell_path.is_absolute():
        return str(shell_path)
    return profile.executable
