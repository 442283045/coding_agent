# AGENTS.md

## Scope

These instructions apply to the whole repository rooted here.

## Project Summary

- This project is an AI-powered CLI coding agent.
- The main package is `coding_agent/`.
- Key areas:
  - `coding_agent/cli.py`: Typer CLI entrypoint.
  - `coding_agent/config.py`: Pydantic settings and environment loading.
  - `coding_agent/agent/`: agent loop and orchestration.
  - `coding_agent/llm/`: provider-facing LLM client code.
  - `coding_agent/tools/`: tool registration and tool implementations.
  - `tests/`: pytest test suite.

## Python Baseline

- Target Python `3.14` for all new code and meaningful refactors.
- Prefer modern Python syntax over legacy compatibility patterns.
- If a change introduces Python 3.12+ syntax that affects repo tooling, update the relevant `pyproject.toml` settings in the same change when appropriate.

## Coding Style

- Prefer `pathlib.Path` over `os.path`.
- Prefer `collections.abc` types such as `Callable`, `Iterable`, `Mapping`, and `Sequence` over legacy `typing` container imports.
- Prefer `X | Y` over `Optional[X]` and `Union[X, Y]`.
- Prefer `type Alias = ...` for type aliases when it improves clarity.
- Use `Self`, `Protocol`, `TypedDict`, `Literal`, `ClassVar`, and `@override` where they make intent clearer.
- Use structural pattern matching (`match`) when it materially improves readability over long dispatch chains.
- Prefer f-strings, comprehensions, context managers, and other modern stdlib patterns instead of manual boilerplate.
- Keep functions focused and side effects explicit.
- Use clear docstrings on public modules, classes, and functions.

## Typing Expectations

- Add type annotations to all new public functions, methods, class attributes, and module-level constants.
- Prefer precise types over `Any`. Treat `Any` as a last resort for external library boundaries.
- Narrow unknown JSON-like data at the boundary instead of passing `dict[str, Any]` deep into the codebase.
- Use small typed helper models for structured data:
  - `BaseModel` for validated external/config payloads.
  - `@dataclass(slots=True)` or small classes for internal structured state when Pydantic is unnecessary.
  - `TypedDict` or `Protocol` for lightweight message or tool payload contracts.
- Keep mypy friendliness in mind even if the current codebase is still catching up.

## Architecture Notes

- Preserve the separation between CLI, agent orchestration, LLM integration, and tool implementations.
- Tool code should stay side-effect aware, validate inputs early, and return user-readable results.
- When changing tool registration or schemas, keep OpenAI-compatible and Anthropic-compatible formatting behavior aligned.
- Prefer adding helper functions over growing already-large methods such as the core agent loop.
- Avoid hidden global state beyond the existing shared settings/registry patterns unless there is a strong reason.

## Async and IO

- Prefer async-first implementations for network, subprocess, and other IO-bound paths.
- Do not introduce blocking work into async paths without isolating it clearly.
- Use explicit encodings for file IO, normally UTF-8.
- Keep filesystem access rooted to the working directory when operating inside tools.

## CLI and UX

- Preserve the Typer + Rich based CLI style already used in the repository.
- Error messages should be actionable and concise.
- Show a visible loading state for operations that may take noticeable time, especially tool execution and MCP configuration or reload flows.
- Do not silently weaken safety checks around shell execution or filesystem boundaries.

## Testing and Verification

- Add or update pytest coverage for behavior changes.
- Favor focused unit tests in `tests/` for:
  - tool registration
  - file and shell safety behavior
  - config parsing
  - LLM message and tool formatting
- Run the most relevant checks after changes:
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run ruff format .`
  - `uv run mypy coding_agent`

## Change Guidelines

- Make small, reviewable changes instead of broad rewrites unless explicitly requested.
- Preserve existing public CLI commands and tool names unless the task requires a breaking change.
- Update `README.md` when user-facing behavior, setup steps, or commands change.
- When touching older code, improve typing and structure opportunistically, but avoid unrelated churn.

## Preferred Patterns

- Prefer:
  - `Path` objects
  - explicit return types
  - narrow exceptions
  - immutable or append-only data flows where practical
  - helper functions for parsing, normalization, and schema conversion
- Avoid:
  - untyped dictionaries passed across multiple layers
  - large monolithic functions growing further
  - compatibility shims for Python versions older than 3.14
  - introducing new dependencies without clear benefit
