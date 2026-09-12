"""Typed application settings, loaded from environment variables / .env.

Nothing secret lives here: the Azure connection is keyless, so the only
configuration is the endpoint URL and the deployment names.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the clinical RAG application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Which model provider to use: real Azure OpenAI, or offline fakes.
    llm_provider: Literal["azure", "mock"] = "azure"

    # Azure OpenAI v1 endpoint, e.g.
    # https://<resource>.services.ai.azure.com/openai/v1/
    azure_openai_base_url: str = ""

    # Deployment names created in the Foundry portal (not model names).
    azure_openai_chat_deployment: str = "chat"
    azure_openai_embeddings_deployment: str = "embeddings"

    # Entra ID scope used to request a bearer token.
    azure_token_scope: str = "https://ai.azure.com/.default"

    # Underlying embedding model, used only for local token counting.
    embedding_model_name: str = "text-embedding-3-small"

    # Where the Chroma indexes live (one subdirectory per provider).
    index_dir: str = ".chroma"

    log_level: str = "INFO"
    audit_log_path: str = "logs/audit.jsonl"

    @model_validator(mode="after")
    def _check_azure_config(self) -> "Settings":
        if self.llm_provider == "azure" and not self.azure_openai_base_url:
            raise ValueError(
                "AZURE_OPENAI_BASE_URL must be set when LLM_PROVIDER=azure. "
                "Copy .env.example to .env and fill in your endpoint."
            )
        if self.azure_openai_base_url and not self.azure_openai_base_url.rstrip(
            "/"
        ).endswith("/openai/v1"):
            raise ValueError(
                "AZURE_OPENAI_BASE_URL should end with /openai/v1/ — for example "
                "https://your-resource.services.ai.azure.com/openai/v1/"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings (cached for the process lifetime)."""
    return Settings()
