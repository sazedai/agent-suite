"""LLM provider setup for Agent Suite."""

from __future__ import annotations

from crewai import LLM
from core.config import settings


def get_llm(provider: str = "openai", model: str | None = None, temperature: float | None = None):
    """Get a configured LLM instance.

    Args:
        provider: ``"openai"`` or ``"anthropic"``
        model: Override the default model name
        temperature: Override the default temperature

    Returns:
        A crewai.LLM instance
    """
    model = model or settings.agent_model
    temperature = temperature if temperature is not None else settings.agent_temperature

    if provider == "anthropic":
        return LLM(
            model=f"anthropic/{model}",
            temperature=temperature,
            max_tokens=settings.agent_max_tokens,
            api_key=settings.anthropic_api_key or None,
        )

    base_url = settings.openai_api_base if settings.openai_api_base != "https://api.openai.com/v1" else None
    return LLM(
        model=f"openai/{model}",
        temperature=temperature,
        max_tokens=settings.agent_max_tokens,
        api_key=settings.openai_api_key or None,
        base_url=base_url,
    )
