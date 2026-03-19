"""Tests for application settings."""

from coding_agent.config import Settings


def test_settings_default_model_uses_moonshot_kimi() -> None:
    """The fallback default model should use the provider-qualified Kimi model name."""
    settings = Settings(_env_file=None)

    assert settings.default_model == "moonshot/kimi-k2.5"


def test_settings_normalize_bare_kimi_model_name() -> None:
    """Bare Kimi model names should be normalized to the Moonshot provider prefix."""
    settings = Settings(_env_file=None, default_model="kimi-k2.5")

    assert settings.default_model == "moonshot/kimi-k2.5"
