"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Annotated

from platformdirs import user_config_dir
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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
    allowed_shell_commands: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["git", "python", "pytest", "ls", "cat", "echo", "find", "grep"]
    )
    max_file_size: int = Field(default=1024 * 1024, ge=1024)  # 1MB default

    # Debug
    debug: bool = Field(default=False)

    @field_validator("allowed_shell_commands", mode="before")
    @classmethod
    def parse_shell_commands(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated shell commands."""
        if isinstance(v, str):
            return [cmd.strip() for cmd in v.split(",") if cmd.strip()]
        return v

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

    def ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
