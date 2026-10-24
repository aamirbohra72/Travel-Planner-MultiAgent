"""Chat-model construction for supported LLM providers."""

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from tripmate.config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Create the configured Groq chat model or report its missing credential."""
    if (
        settings.groq_api_key is None
        or not settings.groq_api_key.get_secret_value().strip()
    ):
        raise ValueError("GROQ_API_KEY is required; set it in .env.")
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key.get_secret_value(),
    )