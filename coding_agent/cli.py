"""CLI entry point using Typer."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from coding_agent.agent.core import Agent
from coding_agent.config import settings

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
ChatPathArgument = Annotated[
    Path,
    typer.Argument(
        help="Working directory for the agent",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
]
RunPromptArgument = Annotated[
    str,
    typer.Argument(help="The task to execute"),
]
RunPathOption = Annotated[
    Path,
    typer.Option(
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
    str,
    typer.Option("--model", "-m", help="LLM model to use"),
]
DebugOption = Annotated[
    bool,
    typer.Option("--debug", "-d", help="Enable debug mode"),
]


def _run_interactive_session(path: Path, model: str, debug: bool) -> None:
    """Start an interactive agent session."""
    # Show welcome banner
    title = Text("Coding Agent", style="bold blue")
    subtitle = Text(f"Model: {model} | Directory: {path}", style="dim")

    console.print(
        Panel(
            subtitle,
            title=title,
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()

    # Initialize and run agent
    try:
        agent = Agent(
            working_dir=str(path),
            model=model,
            debug=debug,
        )
        agent.run_interactive()
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if debug:
            raise
        raise typer.Exit(1) from e


@app.callback()
def main(
    ctx: typer.Context,
    version: VersionOption = None,
) -> None:
    """Coding Agent - AI-powered CLI coding assistant."""
    if ctx.invoked_subcommand is None:
        _run_interactive_session(
            path=DEFAULT_WORKING_DIR.resolve(),
            model=DEFAULT_MODEL,
            debug=DEFAULT_DEBUG,
        )


@app.command()
def chat(
    path: ChatPathArgument = DEFAULT_WORKING_DIR,
    model: ModelOption = DEFAULT_MODEL,
    debug: DebugOption = DEFAULT_DEBUG,
) -> None:
    """Start an interactive coding session."""
    _run_interactive_session(path=path, model=model, debug=debug)


@app.command()
def run(
    prompt: RunPromptArgument,
    path: RunPathOption = DEFAULT_WORKING_DIR,
    model: ModelOption = DEFAULT_MODEL,
    debug: DebugOption = DEFAULT_DEBUG,
) -> None:
    """Execute a single task and exit."""
    try:
        agent = Agent(
            working_dir=str(path),
            model=model,
            debug=debug,
        )
        result = agent.run_once(prompt)
        console.print(result)
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


if __name__ == "__main__":
    app()
