"""Application settings loaded from environment variables and .env."""

from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for TripMate and its MCP server."""

    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="qwen/qwen3.8-27b", validation_alias="GROQ_MODEL"
    )
    tripmate_data_source: Literal["mock", "open-meteo"] = Field(
        default="mock", validation_alias="TRIPMATE_DATA_SOURCE"
    )
    tripmate_default_origin: str = Field(
        default="Delhi (DEL)", validation_alias="TRIPMATE_DEFAULT_ORIGIN"
    )
    tripmate_mcp_host: str = Field(default="127.0.0.1", validation_alias="TRIPMATE_MCP_HOST")
    tripmate_mcp_port: int = Field(
        default=8000, ge=1, le=65535, validation_alias="TRIPMATE_MCP_PORT"
    )
    tripmate_mcp_url: str = Field(
        default="http://127.0.0.1:8000/mcp", validation_alias="TRIPMATE_MCP_URL"
    )
    tripmate_mcp_tool_timeout_seconds: float = Field(
        default=15.0, gt=0, validation_alias="TRIPMATE_MCP_TOOL_TIMEOUT_SECONDS"
    )
    tripmate_database_path: str = Field(
        default="tripmate.sqlite", validation_alias="TRIPMATE_DATABASE_PATH"
    )
    tripmate_profile_id: str = Field(
        default="local-user", validation_alias="TRIPMATE_PROFILE_ID"
    )
    tripmate_max_agent_iterations: int = Field(
        default=12, ge=1, le=50, validation_alias="TRIPMATE_MAX_AGENT_ITERATIONS"
    )
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(
        default=None, validation_alias="LANGSMITH_API_KEY"
    )
    langsmith_project: str = Field(
        default="tripmate", validation_alias="LANGSMITH_PROJECT"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    """Load application settings from the environment and .env file."""
    load_dotenv(override=False)
    return Settings()