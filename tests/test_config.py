"""Tests for application settings."""

from pathlib import Path

from coding_agent.config import MCPServerConfig, Settings


def test_settings_default_model_uses_moonshot_kimi() -> None:
    """The fallback default model should use the provider-qualified Kimi model name."""
    settings = Settings(_env_file=None)

    assert settings.default_model == "moonshot/kimi-k2.5"


def test_settings_normalize_bare_kimi_model_name() -> None:
    """Bare Kimi model names should be normalized to the Moonshot provider prefix."""
    settings = Settings(_env_file=None, default_model="kimi-k2.5")

    assert settings.default_model == "moonshot/kimi-k2.5"


def test_settings_load_mcp_servers_from_json_env() -> None:
    """MCP server config should parse from the environment override."""
    settings = Settings(
        _env_file=None,
        mcp_servers_json=(
            '{"mcpServers":{"filesystem":{"command":"npx","args":["-y","server"],'
            '"env":{"ROOT":"."}}}}'
        ),
    )

    servers = settings.load_mcp_servers()

    assert set(servers) == {"filesystem"}
    assert servers["filesystem"].transport == "stdio"
    assert servers["filesystem"].command == "npx"


def test_settings_save_mcp_servers_writes_json_file(tmp_path: Path) -> None:
    """MCP server config should be writable to the configured JSON file."""

    config_path = tmp_path / "mcp.json"
    settings = Settings(_env_file=None, mcp_config_path=config_path)

    saved_path = settings.save_mcp_servers(
        {
            "filesystem": MCPServerConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "."],
            )
        }
    )

    assert saved_path == config_path.resolve()
    assert config_path.exists()
    assert '"mcpServers"' in config_path.read_text(encoding="utf-8")
    assert settings.load_mcp_servers()["filesystem"].command == "npx"


def test_settings_load_mcp_servers_creates_default_mcp_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Loading MCP servers should create the default config file when it is missing."""

    settings = Settings(_env_file=None)
    monkeypatch.setattr("coding_agent.config.Settings.config_dir", property(lambda self: tmp_path))

    config_path = tmp_path / "mcp.json"
    assert not config_path.exists()

    servers = settings.load_mcp_servers()

    assert servers == {}
    assert config_path.exists()
    assert config_path.read_text(encoding="utf-8") == '{\n  "mcpServers": {}\n}\n'
