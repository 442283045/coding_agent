"""Configuration management using Pydantic Settings."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_model_name(model: str) -> str:
    """Normalize provider-specific shorthand model names."""
    normalized_model = model.strip()
    if "/" in normalized_model:
        return normalized_model

    if normalized_model.lower().startswith(("kimi-", "moonshot-v1")):
        return f"moonshot/{normalized_model}"

    return normalized_model


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENT_",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # LLM API Keys
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    moonshot_api_key: str | None = Field(default=None, alias="MOONSHOT_API_KEY")
    moonshot_api_base: str | None = Field(default=None, alias="MOONSHOT_API_BASE")

    # Model settings
    default_model: str = Field(default="moonshot/kimi-k2.5")
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_iterations: int = Field(default=50, ge=1, le=1000)

    # Safety settings
    max_file_size: int = Field(default=1024 * 1024, ge=1024)  # 1MB default

    # Debug
    debug: bool = Field(default=False)
    mcp_servers_json: str | None = Field(default=None, alias="MCP_SERVERS_JSON")
    mcp_config_path: Path | None = Field(default=None, alias="MCP_CONFIG_PATH")

    @field_validator("default_model")
    @classmethod
    def normalize_default_model(cls, value: str) -> str:
        """Normalize shorthand model names into provider-qualified names."""
        return normalize_model_name(value)

    @property
    def config_dir(self) -> Path:
        """Get user configuration directory."""
        return Path(user_config_dir("coding-agent", "codingagent"))

    @property
    def history_file(self) -> Path:
        """Get path to conversation history file."""
        return self.config_dir / "history.json"

    @property
    def default_mcp_config_path(self) -> Path:
        """Get the default MCP config file path."""
        return self.config_dir / "mcp.json"

    @property
    def effective_mcp_config_path(self) -> Path:
        """Get the MCP config file path used for on-disk persistence."""
        return (self.mcp_config_path or self.default_mcp_config_path).resolve()

    def ensure_default_mcp_config_file(self) -> Path:
        """Create the default MCP config file when it does not already exist."""
        config_path = self.default_mcp_config_path.resolve()
        if config_path.exists():
            return config_path

        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = MCPServersFile.model_validate({"mcpServers": {}})
        config_path.write_text(
            payload.model_dump_json(by_alias=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return config_path

    def load_mcp_servers(self) -> dict[str, MCPServerConfig]:
        """Load MCP server configuration from env or disk."""
        if self.mcp_servers_json:
            raw_config = json.loads(self.mcp_servers_json)
            return MCPServersFile.model_validate(raw_config).mcp_servers

        config_path = (
            self.ensure_default_mcp_config_file()
            if self.mcp_config_path is None
            else self.effective_mcp_config_path
        )
        if not config_path.exists():
            return {}

        content = config_path.read_text(encoding="utf-8").strip()
        if not content:
            return {}

        raw_config = json.loads(content)
        return MCPServersFile.model_validate(raw_config).mcp_servers

    def ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if self.mcp_config_path is None and self.mcp_servers_json is None:
            self.ensure_default_mcp_config_file()

    def save_mcp_servers(self, servers: Mapping[str, MCPServerConfig]) -> Path:
        """Persist MCP server configuration to disk."""
        config_path = self.effective_mcp_config_path
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = MCPServersFile.model_validate({"mcpServers": dict(servers)})
        config_path.write_text(
            payload.model_dump_json(by_alias=True, indent=2),
            encoding="utf-8",
        )
        return config_path

    def update_mcp_servers_json(self, servers: Mapping[str, MCPServerConfig]) -> None:
        """Update the in-memory JSON override for MCP servers."""
        payload = MCPServersFile.model_validate({"mcpServers": dict(servers)})
        self.mcp_servers_json = payload.model_dump_json(by_alias=True, indent=2)


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    transport: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, Any] = Field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    auth: str | None = None
    timeout: int | None = None
    description: str | None = None
    icon: str | None = None
    keep_alive: bool | None = None
    sse_read_timeout: int | float | None = None

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        """Validate that the server shape matches its transport."""
        if self.command:
            if self.transport is None:
                self.transport = "stdio"
            if self.transport != "stdio":
                raise ValueError("command-based MCP servers must use the 'stdio' transport")
            return self

        if self.url:
            if self.transport is None:
                self.transport = "streamable-http"
            if self.transport not in {"http", "streamable-http", "sse"}:
                raise ValueError("remote MCP servers must use http, streamable-http, or sse")
            return self

        raise ValueError("MCP server config must define either 'command' or 'url'")

    def to_fastmcp_config(self) -> dict[str, Any]:
        """Convert to the config shape accepted by FastMCP Client."""
        data = self.model_dump(exclude_none=True)
        if self.command:
            data["transport"] = "stdio"
        return data


class MCPServersFile(BaseModel):
    """Container for MCP server configuration files."""

    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict, alias="mcpServers")

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def wrap_root_servers(cls, value: Any) -> Any:
        """Allow either {'mcpServers': ...} or a bare server mapping."""
        if isinstance(value, dict) and "mcpServers" not in value:
            return {"mcpServers": value}
        return value


# Global settings instance
settings = Settings()
