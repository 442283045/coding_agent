"""Tests for the reusable diagnostics module."""

from pathlib import Path

from coding_agent.config import Settings
from coding_agent.doctor import DoctorReport, build_doctor_report


def test_build_doctor_report_is_ready_for_a_healthy_workspace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A healthy configuration should pass all product-readiness checks."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    mcp_config_path = config_dir / "mcp.json"
    mcp_config_path.write_text('{"mcpServers": {}}', encoding="utf-8")

    settings = Settings(
        _env_file=None,
        default_model="gpt-4o-mini",
        openai_api_key="sk-test",
        mcp_config_path=mcp_config_path,
    )
    monkeypatch.setattr(
        "coding_agent.config.Settings.config_dir",
        property(lambda self: config_dir),
    )

    report = build_doctor_report(settings, working_dir=workspace)

    assert isinstance(report, DoctorReport)
    assert report.working_dir == workspace.resolve()
    assert report.has_failures is False
    assert report.has_warnings is False
    assert all(check.status == "ok" for check in report.checks)


def test_build_doctor_report_flags_missing_provider_credentials(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The doctor should fail when the configured provider is missing credentials."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings = Settings(
        _env_file=None,
        default_model="claude-3-7-sonnet",
        anthropic_api_key=None,
    )
    monkeypatch.setattr(
        "coding_agent.config.Settings.config_dir",
        property(lambda self: config_dir),
    )

    report = build_doctor_report(settings, working_dir=workspace)
    credentials_check = report.get_check("provider_credentials")

    assert report.has_failures is True
    assert credentials_check.status == "fail"
    assert "ANTHROPIC_API_KEY" in credentials_check.summary


def test_build_doctor_report_fails_on_invalid_mcp_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Invalid MCP configuration should be reported as a blocking failure."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    invalid_mcp_path = config_dir / "mcp.json"
    invalid_mcp_path.write_text("{", encoding="utf-8")

    settings = Settings(_env_file=None, mcp_config_path=invalid_mcp_path)
    monkeypatch.setattr(
        "coding_agent.config.Settings.config_dir",
        property(lambda self: config_dir),
    )

    report = build_doctor_report(settings, working_dir=workspace)
    mcp_check = report.get_check("mcp_config")

    assert report.has_failures is True
    assert mcp_check.status == "fail"
    assert "could not be parsed" in mcp_check.summary.lower()


def test_build_doctor_report_warns_when_provider_is_unknown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Custom model names should produce a warning instead of a false positive."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings = Settings(_env_file=None, default_model="custom-model")
    monkeypatch.setattr(
        "coding_agent.config.Settings.config_dir",
        property(lambda self: config_dir),
    )

    report = build_doctor_report(settings, working_dir=workspace)
    model_check = report.get_check("default_model")
    credentials_check = report.get_check("provider_credentials")

    assert report.has_failures is False
    assert report.has_warnings is True
    assert model_check.status == "warn"
    assert credentials_check.status == "warn"


def test_build_doctor_report_fails_for_unwritable_workspaces(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Workspace write failures should surface as a blocking doctor error."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings = Settings(_env_file=None, openai_api_key="sk-test", default_model="gpt-4o-mini")
    monkeypatch.setattr(
        "coding_agent.config.Settings.config_dir",
        property(lambda self: config_dir),
    )

    def fail_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        raise PermissionError("denied")

    monkeypatch.setattr("coding_agent.doctor.tempfile.mkstemp", fail_mkstemp)

    report = build_doctor_report(settings, working_dir=workspace)
    workspace_check = report.get_check("workspace")

    assert report.has_failures is True
    assert workspace_check.status == "fail"
    assert "not writable" in workspace_check.summary.lower()
