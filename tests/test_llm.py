"""Chat-model provider configuration tests."""

from typing import Any

import pytest

from tripmate.config import Settings
from tripmate import llm


def test_create_chat_model_uses_groq_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_chat_groq(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm, "ChatGroq", fake_chat_groq)
    settings = Settings(
        _env_file=None,
        TRIPMATE_LLM_PROVIDER="groq",
        GROQ_API_KEY="test-groq-key",
        GROQ_MODEL="test-groq-model",
    )

    llm.create_chat_model(settings)

    assert captured == {"model": "test-groq-model", "api_key": "test-groq-key"}


def test_settings_default_to_available_groq_model() -> None:
    settings = Settings(_env_file=None)

    assert settings.groq_model == "qwen/qwen3.8-27b"


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_create_chat_model_requires_nonempty_provider_key(
    api_key: str | None,
) -> None:
    settings = Settings(_env_file=None, GROQ_API_KEY=api_key)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        llm.create_chat_model(settings)