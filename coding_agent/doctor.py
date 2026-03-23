"""Product-readiness diagnostics for the CLI environment."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from coding_agent.agent.history import SessionStore
from coding_agent.config import Settings, normalize_model_name

type DoctorStatus = Literal["ok", "warn", "fail"]


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """A single diagnostic check with a user-facing outcome."""

    key: str
    label: str
    status: DoctorStatus
    summary: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """The aggregated environment diagnostics for one workspace."""

    working_dir: Path
    checks: tuple[DoctorCheck, ...]

    @property
    def has_failures(self) -> bool:
        """Return whether any required checks failed."""
        return any(check.status == "fail" for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        """Return whether any checks produced warnings."""
        return any(check.status == "warn" for check in self.checks)

    def get_check(self, key: str) -> DoctorCheck:
        """Look up one check by key."""
        for check in self.checks:
            if check.key == key:
                return check
        raise KeyError(key)


def build_doctor_report(
    settings_obj: Settings,
    *,
    working_dir: Path,
) -> DoctorReport:
    """Run the standard environment diagnostics for the current workspace."""
    resolved_working_dir = working_dir.resolve()
    checks = (
        _build_model_check(settings_obj),
        _build_provider_credentials_check(settings_obj),
        _build_config_dir_check(settings_obj),
        _build_sessions_dir_check(settings_obj),
        _build_mcp_check(settings_obj),
        _build_workspace_check(resolved_working_dir),
    )
    return DoctorReport(working_dir=resolved_working_dir, checks=checks)


def _build_model_check(settings_obj: Settings) -> DoctorCheck:
    """Validate the configured default model and infer its provider."""
    configured_model = settings_obj.default_model
    normalized_model = normalize_model_name(configured_model)
    provider = _infer_provider(normalized_model)

    if provider is None:
        return DoctorCheck(
            key="default_model",
            label="Default model",
            status="warn",
            summary=f"Configured default model is `{normalized_model}`.",
            detail=(
                "The provider could not be inferred automatically. Make sure the model name "
                "matches the credentials you configured."
            ),
        )

    detail = f"Inferred provider: `{provider}`."
    if configured_model != normalized_model:
        detail = f"Normalized to `{normalized_model}`. {detail}"

    return DoctorCheck(
        key="default_model",
        label="Default model",
        status="ok",
        summary=f"Configured default model is `{normalized_model}`.",
        detail=detail,
    )


def _build_provider_credentials_check(settings_obj: Settings) -> DoctorCheck:
    """Check that the active model has matching credentials configured."""
    model_name = normalize_model_name(settings_obj.default_model)
    provider = _infer_provider(model_name)

    if provider is None:
        return DoctorCheck(
            key="provider_credentials",
            label="Provider credentials",
            status="warn",
            summary=f"Could not determine which API key `{model_name}` needs.",
            detail=(
                "Use a provider-qualified model name or verify the matching credentials "
                "manually."
            ),
        )

    credential_name, credential_value = _credential_for_provider(settings_obj, provider)
    if credential_value:
        return DoctorCheck(
            key="provider_credentials",
            label="Provider credentials",
            status="ok",
            summary=f"Found `{credential_name}` for `{model_name}`.",
        )

    return DoctorCheck(
        key="provider_credentials",
        label="Provider credentials",
        status="fail",
        summary=f"`{model_name}` requires `{credential_name}`, but it is not configured.",
        detail=(
            "Add the missing API key in `.env` or your shell environment before starting "
            "a session."
        ),
    )


def _build_config_dir_check(settings_obj: Settings) -> DoctorCheck:
    """Ensure the user config directory can be created and used."""
    try:
        settings_obj.ensure_config_dir()
    except Exception as exc:
        return DoctorCheck(
            key="config_dir",
            label="Config directory",
            status="fail",
            summary=f"Could not initialize `{settings_obj.config_dir}`.",
            detail=str(exc),
        )

    return DoctorCheck(
        key="config_dir",
        label="Config directory",
        status="ok",
        summary=f"Config directory is ready at `{settings_obj.config_dir}`.",
    )


def _build_sessions_dir_check(settings_obj: Settings) -> DoctorCheck:
    """Ensure the persisted session store can be created."""
    store = SessionStore(settings_obj)
    try:
        sessions_dir = store.ensure_sessions_dir()
    except Exception as exc:
        return DoctorCheck(
            key="sessions_dir",
            label="Session store",
            status="fail",
            summary="Could not initialize the saved-session directory.",
            detail=str(exc),
        )

    return DoctorCheck(
        key="sessions_dir",
        label="Session store",
        status="ok",
        summary=f"Saved sessions are available under `{sessions_dir}`.",
    )


def _build_mcp_check(settings_obj: Settings) -> DoctorCheck:
    """Ensure the current MCP configuration source is loadable."""
    try:
        servers = settings_obj.load_mcp_servers()
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        return DoctorCheck(
            key="mcp_config",
            label="MCP configuration",
            status="fail",
            summary="MCP configuration could not be parsed.",
            detail=str(exc),
        )
    except Exception as exc:
        return DoctorCheck(
            key="mcp_config",
            label="MCP configuration",
            status="fail",
            summary="MCP configuration could not be loaded.",
            detail=str(exc),
        )

    source = (
        "`AGENT_MCP_SERVERS_JSON`"
        if settings_obj.mcp_servers_json is not None
        else f"`{settings_obj.effective_mcp_config_path}`"
    )
    return DoctorCheck(
        key="mcp_config",
        label="MCP configuration",
        status="ok",
        summary=f"Loaded {len(servers)} MCP server(s) from {source}.",
    )


def _build_workspace_check(working_dir: Path) -> DoctorCheck:
    """Ensure the requested workspace exists and is writable."""
    if not working_dir.exists():
        return DoctorCheck(
            key="workspace",
            label="Workspace",
            status="fail",
            summary=f"Workspace `{working_dir}` does not exist.",
        )

    if not working_dir.is_dir():
        return DoctorCheck(
            key="workspace",
            label="Workspace",
            status="fail",
            summary=f"`{working_dir}` is not a directory.",
        )

    try:
        next(working_dir.iterdir(), None)
    except PermissionError as exc:
        return DoctorCheck(
            key="workspace",
            label="Workspace",
            status="fail",
            summary=f"Workspace `{working_dir}` is not readable.",
            detail=str(exc),
        )

    try:
        descriptor, temp_path = tempfile.mkstemp(
            prefix=".coding-agent-doctor-",
            dir=working_dir,
            text=True,
        )
        os.close(descriptor)
        Path(temp_path).unlink(missing_ok=True)
    except PermissionError as exc:
        return DoctorCheck(
            key="workspace",
            label="Workspace",
            status="fail",
            summary=f"Workspace `{working_dir}` is not writable.",
            detail=str(exc),
        )
    except Exception as exc:
        return DoctorCheck(
            key="workspace",
            label="Workspace",
            status="fail",
            summary=f"Could not write a temporary file inside `{working_dir}`.",
            detail=str(exc),
        )

    return DoctorCheck(
        key="workspace",
        label="Workspace",
        status="ok",
        summary=f"Workspace `{working_dir}` is readable and writable.",
    )


def _infer_provider(model_name: str) -> str | None:
    """Infer a provider from a normalized model name."""
    normalized = model_name.strip().lower()
    if normalized.startswith("moonshot/") or normalized.startswith("kimi-") or "kimi" in normalized:
        return "moonshot"
    if normalized.startswith("anthropic/") or "claude" in normalized:
        return "anthropic"
    if normalized.startswith("openai/"):
        return "openai"
    if normalized.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "openai"
    return None


def _credential_for_provider(settings_obj: Settings, provider: str) -> tuple[str, str | None]:
    """Return the required credential name and current value for a provider."""
    match provider:
        case "openai":
            return "OPENAI_API_KEY", settings_obj.openai_api_key
        case "anthropic":
            return "ANTHROPIC_API_KEY", settings_obj.anthropic_api_key
        case "moonshot":
            return "MOONSHOT_API_KEY", settings_obj.moonshot_api_key
        case _:
            raise ValueError(f"Unsupported provider: {provider}")
