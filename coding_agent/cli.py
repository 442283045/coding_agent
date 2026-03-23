"""CLI entry point using Typer."""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from coding_agent.agent.core import Agent
from coding_agent.agent.history import SessionNotFoundError, SessionStore
from coding_agent.config import settings
from coding_agent.doctor import build_doctor_report

app = typer.Typer(
    help="AI-powered CLI coding agent",
    add_completion=True,
    rich_markup_mode="rich",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()
DEFAULT_WORKING_DIR = Path(".")
DEFAULT_MODEL = settings.default_model
DEFAULT_DEBUG = settings.debug
sessions_app = typer.Typer(help="Inspect saved agent sessions.")
app.add_typer(sessions_app, name="sessions")


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        from coding_agent import __version__

        console.print(f"[bold blue]coding-agent[/bold blue] version {__version__}")
        raise typer.Exit()


VersionOption = Annotated[
    bool | None,
    typer.Option("--version", "-v", callback=version_callback, is_eager=True),
]
WorkspaceOption = Annotated[
    Path | None,
    typer.Option(
        "--workspace",
        "-w",
        help="Working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=False,
    ),
]
ResumeOption = Annotated[
    str | None,
    typer.Option("--resume", "-r", help="Resume a saved session by session id."),
]
ChatPathArgument = Annotated[
    Path | None,
    typer.Argument(
        help="Working directory for the agent",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=False,
    ),
]
RunPromptArgument = Annotated[
    str,
    typer.Argument(help="The task to execute"),
]
RunPathOption = Annotated[
    Path | None,
    typer.Option(
        "--workspace",
        "-w",
        "--path",
        "-p",
        help="Working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
]
ModelOption = Annotated[
    str | None,
    typer.Option("--model", "-m", help="LLM model to use"),
]
DebugOption = Annotated[
    bool,
    typer.Option("--debug", "-d", help="Enable debug mode"),
]
SessionIdArgument = Annotated[
    str,
    typer.Argument(help="Saved session id"),
]
SessionNameArgument = Annotated[
    str | None,
    typer.Argument(help="New explicit session name. Quote it when it contains spaces."),
]
ExportFormatOption = Annotated[
    str,
    typer.Option("--format", "-f", help="Export format: markdown or json."),
]
ExportOutputOption = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        help="Write the export to a file instead of stdout.",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
]
ClearNameOption = Annotated[
    bool,
    typer.Option(
        "--clear",
        help="Clear the explicit session name and fall back to the first user message.",
    ),
]


def _resolve_working_dir(
    *,
    path: Path | None = None,
    workspace: Path | None = None,
) -> Path:
    """Resolve the working directory from path argument and workspace option."""
    if path is not None and workspace is not None and path.resolve() != workspace.resolve():
        raise typer.BadParameter("Use either the path argument or --workspace/-w, not both.")

    if workspace is not None:
        return workspace.resolve()
    if path is not None:
        return path.resolve()
    return DEFAULT_WORKING_DIR.resolve()


def _validate_resume_inputs(
    *,
    resume_session_id: str | None,
    path: Path | None = None,
    workspace: Path | None = None,
) -> None:
    """Reject conflicting resume and workspace inputs."""
    if resume_session_id is not None and (path is not None or workspace is not None):
        raise typer.BadParameter("Use either a workspace path or --resume/-r, not both.")


def _create_agent(
    *,
    working_dir: Path | None,
    model: str | None,
    debug: bool,
    resume_session_id: str | None,
) -> Agent:
    """Create an agent for a new or resumed session."""
    return Agent(
        working_dir=str(working_dir) if working_dir is not None else None,
        model=model,
        debug=debug,
        resume_session_id=resume_session_id,
    )


def _resolve_requested_model(
    *,
    model: str | None,
    resume_session_id: str | None,
) -> str | None:
    """Preserve the saved model on resume unless the user overrides it explicitly."""
    if resume_session_id is not None:
        return model
    return model or DEFAULT_MODEL


def _run_interactive_session(
    *,
    working_dir: Path | None,
    workspace_display: Path | None,
    model: str | None,
    debug: bool,
    resume_session_id: str | None = None,
) -> None:
    """Start an interactive agent session."""
    agent = _create_agent(
        working_dir=working_dir,
        model=model,
        debug=debug,
        resume_session_id=resume_session_id,
    )

    # Show welcome banner
    title = Text("Coding Agent", style="bold blue")
    displayed_workspace = workspace_display or agent.working_dir
    subtitle = Text(
        f"Model: {agent.model} | Workspace: {displayed_workspace} | Session: {agent.session_id}",
        style="dim",
    )

    console.print(
        Panel(
            subtitle,
            title=title,
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()

    try:
        agent.run_interactive()
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
    except SessionNotFoundError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if debug:
            raise
        raise typer.Exit(1) from e
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()


def _get_workspace_display(
    *,
    path: Path | None = None,
    workspace: Path | None = None,
) -> Path:
    """Get the user-facing workspace path for the banner."""
    if workspace is not None:
        return workspace
    if path is not None:
        return path
    return DEFAULT_WORKING_DIR.resolve()


@app.callback()
def main(
    ctx: typer.Context,
    version: VersionOption = None,
    workspace: WorkspaceOption = None,
    resume: ResumeOption = None,
) -> None:
    """Coding Agent - AI-powered CLI coding assistant."""
    if ctx.invoked_subcommand is None:
        _validate_resume_inputs(resume_session_id=resume, workspace=workspace)
        _run_interactive_session(
            working_dir=None if resume is not None else _resolve_working_dir(workspace=workspace),
            workspace_display=(
                None if resume is not None else _get_workspace_display(workspace=workspace)
            ),
            model=_resolve_requested_model(model=None, resume_session_id=resume),
            debug=DEFAULT_DEBUG,
            resume_session_id=resume,
        )


@app.command()
def chat(
    path: ChatPathArgument = None,
    workspace: WorkspaceOption = None,
    resume: ResumeOption = None,
    model: ModelOption = None,
    debug: DebugOption = DEFAULT_DEBUG,
) -> None:
    """Start an interactive coding session."""
    _validate_resume_inputs(resume_session_id=resume, path=path, workspace=workspace)
    working_dir = (
        None if resume is not None else _resolve_working_dir(path=path, workspace=workspace)
    )
    _run_interactive_session(
        working_dir=working_dir,
        workspace_display=(
            None if resume is not None else _get_workspace_display(path=path, workspace=workspace)
        ),
        model=_resolve_requested_model(model=model, resume_session_id=resume),
        debug=debug,
        resume_session_id=resume,
    )


@app.command()
def run(
    prompt: RunPromptArgument,
    path: RunPathOption = None,
    resume: ResumeOption = None,
    model: ModelOption = None,
    debug: DebugOption = DEFAULT_DEBUG,
) -> None:
    """Execute a single task and exit."""
    try:
        _validate_resume_inputs(resume_session_id=resume, path=path)
        working_dir = None if resume is not None else _resolve_working_dir(path=path)
        agent = _create_agent(
            working_dir=working_dir,
            model=_resolve_requested_model(model=model, resume_session_id=resume),
            debug=debug,
            resume_session_id=resume,
        )
        result = agent.run_once(prompt)
        console.print(Markdown(result))
    except SessionNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if debug:
            raise
        raise typer.Exit(1) from e


@app.command()
def config() -> None:
    """Show current configuration."""
    from rich.table import Table

    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Default Model", settings.default_model)
    table.add_row("Max Tokens", str(settings.max_tokens))
    table.add_row("Temperature", str(settings.temperature))
    table.add_row("Max Iterations", str(settings.max_iterations))
    table.add_row("Config Directory", str(settings.config_dir))
    table.add_row("Debug Mode", str(settings.debug))
    table.add_row(
        "OpenAI API Key",
        "[green]✓[/green]" if settings.openai_api_key else "[red]✗[/red]",
    )
    table.add_row(
        "Anthropic API Key",
        "[green]✓[/green]" if settings.anthropic_api_key else "[red]✗[/red]",
    )

    console.print(table)


def _format_doctor_status(status: str) -> str:
    """Render a doctor status label with Rich styling."""
    match status:
        case "ok":
            return "[green]OK[/green]"
        case "warn":
            return "[yellow]WARN[/yellow]"
        case _:
            return "[red]FAIL[/red]"


@app.command()
def doctor(workspace: WorkspaceOption = None) -> None:
    """Run environment self-checks for the current workspace."""
    working_dir = _resolve_working_dir(workspace=workspace)
    report = build_doctor_report(settings, working_dir=working_dir)

    overall_status = "FAIL" if report.has_failures else "WARN" if report.has_warnings else "OK"
    console.print(
        f"[bold blue]Coding Agent Doctor[/bold blue] {overall_status} "
        f"[dim]({report.working_dir})[/dim]"
    )

    table = Table()
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="white", no_wrap=True)
    table.add_column("Summary", style="white", overflow="fold")
    table.add_column("Detail", style="dim", overflow="fold")

    for check in report.checks:
        table.add_row(
            check.label,
            _format_doctor_status(check.status),
            check.summary,
            check.detail or "",
        )

    console.print(table)

    if report.has_failures:
        raise typer.Exit(1)


def _get_session_store() -> SessionStore:
    """Create the default session store used by CLI commands."""
    return SessionStore(settings)


def _format_session_timestamp(value: object) -> str:
    """Render a compact local timestamp for session tables."""
    if isinstance(value, datetime):
        return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _build_session_export_content(
    *,
    session_id: str,
    format_name: str,
    store: SessionStore,
) -> str:
    """Render a saved session in the requested export format."""
    match format_name.strip().lower():
        case "markdown" | "md":
            return store.export_session_markdown(session_id)
        case "json":
            return (
                json.dumps(
                    store.export_session_json(session_id),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        case _:
            raise typer.BadParameter("Export format must be one of: markdown, md, json.")


@sessions_app.command("list")
def list_sessions() -> None:
    """List saved chat sessions."""
    store = _get_session_store()
    sessions = store.list_sessions()

    if not sessions:
        console.print("[yellow]No saved sessions found.[/yellow]")
        return

    table = Table(title="Saved Sessions")
    table.add_column("Name", style="white", overflow="fold")
    table.add_column("Session", style="cyan")
    table.add_column("Updated", style="green")
    table.add_column("Messages", justify="right")
    table.add_column("Model", style="magenta")
    table.add_column("Workspace", overflow="fold")

    for session in sessions:
        table.add_row(
            session.display_name,
            session.session_id,
            _format_session_timestamp(session.updated_at),
            str(session.message_count),
            session.model,
            str(session.working_dir),
        )

    console.print(table)


@sessions_app.command("show")
def show_session(session_id: SessionIdArgument) -> None:
    """Show details for one saved chat session."""
    store = _get_session_store()
    try:
        session = store.load_session(session_id)
    except SessionNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    lines = [
        f"# Session `{session.session_id}`",
        "",
        f"- Name: {session.display_name}",
        f"- Workspace: `{session.working_dir}`",
        f"- Model: `{session.model}`",
        f"- Created: `{_format_session_timestamp(session.created_at)}`",
        f"- Updated: `{_format_session_timestamp(session.updated_at)}`",
        f"- Messages: `{session.message_count}`",
    ]

    if session.name is not None:
        lines.append(f"- Explicit name: `{session.name}`")
    if session.interaction_log_path is not None:
        lines.append(f"- LLM log: `{session.interaction_log_path}`")
    if session.first_user_message:
        lines.append(f"- First user message: {session.first_user_message}")
    if session.last_user_message:
        lines.append(f"- Last user message: {session.last_user_message}")

    lines.extend(
        [
            "",
            "Resume with:",
            f"- `coding-agent chat --resume {session.session_id}`",
        ]
    )

    console.print(Markdown("\n".join(lines)))


@sessions_app.command("rename")
def rename_session(
    session_id: SessionIdArgument,
    name: SessionNameArgument = None,
    clear: ClearNameOption = False,
) -> None:
    """Set or clear an explicit saved-session name."""
    if clear and name is not None:
        raise typer.BadParameter("Use either a name argument or --clear, not both.")
    if not clear and name is None:
        raise typer.BadParameter("Provide a new name or use --clear.")

    store = _get_session_store()
    try:
        session = (
            store.clear_session_name(session_id)
            if clear
            else store.rename_session(session_id, name or "")
        )
    except SessionNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    action = "Cleared explicit session name" if clear else "Renamed session"
    console.print(
        Markdown(
            "\n".join(
                [
                    f"{action} `{session.session_id}`.",
                    "",
                    f"- Name: {session.display_name}",
                ]
            )
        )
    )


@sessions_app.command("export")
def export_session(
    session_id: SessionIdArgument,
    format_name: ExportFormatOption = "markdown",
    output: ExportOutputOption = None,
) -> None:
    """Export one saved session as Markdown or JSON."""
    store = _get_session_store()
    try:
        exported = _build_session_export_content(
            session_id=session_id,
            format_name=format_name,
            store=store,
        )
    except SessionNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(exported, encoding="utf-8")
        console.print(f"[green]Exported session to {output.resolve()}[/green]")
        return

    console.print(exported, markup=False, highlight=False, soft_wrap=True)


if __name__ == "__main__":
    app()
