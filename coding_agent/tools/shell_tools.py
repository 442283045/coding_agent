"""Shell execution tools for the agent."""

import asyncio
from pathlib import Path
from typing import Any

from coding_agent.shell_environment import detect_shell_profile, resolve_shell_executable
from coding_agent.tools.registry import registry


def _get_working_dir(ctx: dict[str, Any] | None) -> Path:
    """Get working directory from context or use current directory."""
    if ctx and "working_dir" in ctx:
        return Path(ctx["working_dir"]).resolve()
    return Path.cwd()


def _validate_command(command: str) -> tuple[bool, str]:
    """Validate a shell command before execution."""
    if not command.strip():
        return False, "Empty command"

    # Keep a coarse catastrophic-pattern blocklist even when arbitrary commands are allowed.
    dangerous_patterns = [
        "rm -rf /",
        "rm -rf /*",
        "> /",
        "dd if=",
        "mkfs",
        ":(){ :|:& };:",  # Fork bomb
        "chmod -R 777 /",
    ]

    for pattern in dangerous_patterns:
        if pattern in command:
            return False, f"Dangerous pattern detected: {pattern}"

    return True, ""


async def _run_shell_command(
    command: str,
    *,
    working_dir: Path,
    timeout: int,
) -> str:
    """Execute one command in the detected system shell."""
    shell_profile = detect_shell_profile()
    process = await asyncio.create_subprocess_exec(
        resolve_shell_executable(shell_profile),
        *shell_profile.arguments,
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(working_dir),
        limit=1024 * 1024,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return f"Error: Command timed out after {timeout} seconds"

    stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
    stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

    result_parts: list[str] = []

    if process.returncode != 0:
        result_parts.append(f"Exit code: {process.returncode}")

    if stdout_str:
        if len(stdout_str) > 10000:
            stdout_str = stdout_str[:10000] + "\n... [truncated]"
        result_parts.append(f"stdout:\n```\n{stdout_str}\n```")

    if stderr_str:
        if len(stderr_str) > 5000:
            stderr_str = stderr_str[:5000] + "\n... [truncated]"
        result_parts.append(f"stderr:\n```\n{stderr_str}\n```")

    if not result_parts:
        return "Command executed successfully (no output)."

    return "\n\n".join(result_parts)


@registry.tool(
    name="execute_shell",
    description="Execute a shell command in the user's native shell environment. "
    "Windows sessions use PowerShell; macOS and Linux sessions use bash. "
    "Returns stdout and stderr output.",
)
async def execute_shell(
    command: str,
    timeout: int = 30,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Execute a shell command with safety checks."""
    working_dir = _get_working_dir(ctx)

    # Validate command
    allowed, reason = _validate_command(command)
    if not allowed:
        return f"Error: {reason}"

    try:
        return await _run_shell_command(command, working_dir=working_dir, timeout=timeout)
    except Exception as e:
        return f"Error executing command: {e}"


@registry.tool(
    name="view_git_log",
    description="View recent git commit history. Shows commit hash, author, date, and message.",
)
async def view_git_log(
    count: int = 10,
    ctx: dict[str, Any] | None = None,
) -> str:
    """View recent git commits."""
    working_dir = _get_working_dir(ctx)

    try:
        result = await _run_shell_command(
            f'git -C "{working_dir}" log --oneline -n {count}',
            working_dir=working_dir,
            timeout=30,
        )
        if result.startswith("Exit code:") or result.startswith("Error:"):
            return result
        if "stdout:" not in result:
            return "No git commits found or not a git repository."
        return f"Recent commits:\n{result.removeprefix('stdout:\n')}"
    except Exception as e:
        return f"Error viewing git log: {e}"


@registry.tool(
    name="view_git_status",
    description="View git status - shows modified, staged, and untracked files.",
)
async def view_git_status(
    ctx: dict[str, Any] | None = None,
) -> str:
    """View git status."""
    working_dir = _get_working_dir(ctx)

    try:
        result = await _run_shell_command(
            f'git -C "{working_dir}" status',
            working_dir=working_dir,
            timeout=30,
        )
        if result.startswith("Exit code:") or result.startswith("Error:"):
            return result
        return f"Git status:\n{result.removeprefix('stdout:\n')}"
    except Exception as e:
        return f"Error viewing git status: {e}"
