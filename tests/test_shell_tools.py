"""Tests for shell execution behavior."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_agent.shell_environment import detect_shell_profile
from coding_agent.tools import shell_tools


def test_detect_shell_profile_prefers_powershell_on_windows(monkeypatch) -> None:
    """Windows sessions should use PowerShell semantics."""
    monkeypatch.setattr("coding_agent.shell_environment.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "coding_agent.shell_environment.shutil.which",
        lambda name: "C:\\Program Files\\PowerShell\\7\\pwsh.exe" if name == "pwsh" else None,
    )

    profile = detect_shell_profile()

    assert profile.operating_system == "Windows"
    assert profile.shell_name == "PowerShell"
    assert profile.arguments == ("-NoLogo", "-NoProfile", "-Command")


def test_detect_shell_profile_uses_bash_on_linux(monkeypatch) -> None:
    """Linux sessions should use bash semantics."""
    monkeypatch.setattr("coding_agent.shell_environment.platform.system", lambda: "Linux")
    monkeypatch.setattr("coding_agent.shell_environment.shutil.which", lambda name: "/usr/bin/bash")

    profile = detect_shell_profile()

    assert profile.operating_system == "Linux"
    assert profile.shell_name == "bash"
    assert profile.arguments == ("-lc",)


@pytest.mark.asyncio
async def test_execute_shell_allows_non_whitelisted_command(monkeypatch, tmp_path: Path) -> None:
    """Arbitrary commands should execute as long as they do not match the blocklist."""

    captured: dict[str, object] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"ok", b"")

        def kill(self) -> None:
            return None

        async def wait(self) -> None:
            return None

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        shell_tools,
        "detect_shell_profile",
        lambda: SimpleNamespace(
            executable="powershell",
            arguments=("-NoLogo", "-NoProfile", "-Command"),
        ),
    )
    monkeypatch.setattr(
        "coding_agent.tools.shell_tools.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    result = await shell_tools.execute_shell("npm test", ctx={"working_dir": str(tmp_path)})

    assert result == "stdout:\n```\nok\n```"
    assert captured["args"] == ("powershell", "-NoLogo", "-NoProfile", "-Command", "npm test")
    assert captured["kwargs"]["cwd"] == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_execute_shell_blocks_catastrophic_patterns(tmp_path: Path) -> None:
    """Catastrophic command patterns should still be rejected."""

    result = await shell_tools.execute_shell("rm -rf /", ctx={"working_dir": str(tmp_path)})

    assert "Dangerous pattern detected" in result
