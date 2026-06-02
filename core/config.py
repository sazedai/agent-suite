"""Core configuration for Agent Suite."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_base: str = Field(default="https://api.openai.com/v1", alias="OPENAI_API_BASE")

    # Agent
    agent_model: str = Field(default="gpt-4o", alias="AGENT_MODEL")
    agent_temperature: float = Field(default=0.7, alias="AGENT_TEMPERATURE")
    agent_max_tokens: int = Field(default=4096, alias="AGENT_MAX_TOKENS")

    # Storage
    database_url: str = Field(
        default="sqlite+aiosqlite:///./agentsuite.db", alias="DATABASE_URL"
    )

    # Integrations
    hubspot_api_key: str = Field(default="", alias="HUBSPOT_API_KEY")
    shopify_api_key: str = Field(default="", alias="SHOPIFY_API_KEY")
    shopify_api_secret: str = Field(default="", alias="SHOPIFY_API_SECRET")
    shopify_store_url: str = Field(default="", alias="SHOPIFY_STORE_URL")
    gmail_credentials_path: str = Field(default="", alias="GMAIL_CREDENTIALS_PATH")
    google_calendar_credentials_path: str = Field(
        default="", alias="GOOGLE_CALENDAR_CREDENTIALS_PATH"
    )
    slack_bot_token: str = Field(default="", alias="SLACK_BOT_TOKEN")
    stripe_api_key: str = Field(default="", alias="STRIPE_API_KEY")
    semrush_api_key: str = Field(default="", alias="SEMRUSH_API_KEY")
    ahrefs_api_key: str = Field(default="", alias="AHREFS_API_KEY")

    # Search
    serper_api_key: str = Field(default="", alias="SERPER_API_KEY")
    brave_search_api_key: str = Field(default="", alias="BRAVE_SEARCH_API_KEY")


settings = Settings()
