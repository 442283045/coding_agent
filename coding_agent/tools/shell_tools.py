"""Shell execution tools for the agent."""

import asyncio
import shlex
from pathlib import Path
from typing import Any

from coding_agent.config import settings
from coding_agent.tools.registry import registry


def _get_working_dir(ctx: dict[str, Any] | None) -> Path:
    """Get working directory from context or use current directory."""
    if ctx and "working_dir" in ctx:
        return Path(ctx["working_dir"]).resolve()
    return Path.cwd()


def _is_command_allowed(command: str) -> tuple[bool, str]:
    """Check if command is in the allowed list."""
    # Parse the command to get the base command
    try:
        parts = shlex.split(command)
        if not parts:
            return False, "Empty command"
        base_cmd = parts[0]
    except ValueError as e:
        return False, f"Invalid command syntax: {e}"

    # Handle common aliases
    alias_map = {
        "ll": "ls",
        "la": "ls",
        "py": "python",
        "py3": "python",
        "pytest": "pytest",
    }
    base_cmd = alias_map.get(base_cmd, base_cmd)

    allowed = settings.allowed_shell_commands
    if base_cmd not in allowed:
        return False, f"Command '{base_cmd}' is not allowed. Allowed: {', '.join(allowed)}"

    # Block dangerous patterns
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


@registry.tool(
    name="execute_shell",
    description="Execute a shell command safely. Only pre-approved commands are allowed. "
    "Common allowed commands: git, python, pytest, ls, cat, echo, find, grep. "
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
    allowed, reason = _is_command_allowed(command)
    if not allowed:
        return f"Error: {reason}"

    try:
        # Execute command with timeout
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(working_dir),
            limit=1024 * 1024,  # 1MB buffer limit
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

        # Decode output
        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        # Build result
        result_parts = []

        if process.returncode != 0:
            result_parts.append(f"Exit code: {process.returncode}")

        if stdout_str:
            # Truncate if too long
            if len(stdout_str) > 10000:
                stdout_str = stdout_str[:10000] + "\n... [truncated]"
            result_parts.append(f"stdout:\n```\n{stdout_str}\n```")

        if stderr_str:
            # Truncate if too long
            if len(stderr_str) > 5000:
                stderr_str = stderr_str[:5000] + "\n... [truncated]"
            result_parts.append(f"stderr:\n```\n{stderr_str}\n```")

        if not result_parts:
            return "Command executed successfully (no output)."

        return "\n\n".join(result_parts)

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

    command = f'git -C "{working_dir}" log --oneline -n {count}'

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            stderr_str = stderr.decode("utf-8", errors="replace")
            return f"Error: {stderr_str}"

        stdout_str = stdout.decode("utf-8", errors="replace")
        if not stdout_str.strip():
            return "No git commits found or not a git repository."

        return f"Recent commits:\n```\n{stdout_str}```"

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

    command = f'git -C "{working_dir}" status'

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            stderr_str = stderr.decode("utf-8", errors="replace")
            return f"Error: {stderr_str}"

        stdout_str = stdout.decode("utf-8", errors="replace")
        return f"Git status:\n```\n{stdout_str}```"

    except Exception as e:
        return f"Error viewing git status: {e}"
