"""Chat and embedding model clients.

Two providers:

* ``azure`` - Azure OpenAI through its v1 API, authenticated with Microsoft
  Entra ID. No API keys exist anywhere in this project: a short-lived bearer
  token is fetched per request from the signed-in identity (``az login``
  locally, a managed identity when deployed).
* ``mock``  - deterministic offline fakes, so tests and demos run with no
  network access and no Azure subscription.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from .config import Settings, get_settings

MOCK_ANSWER = (
    "[mock] Coverage requires 6 weeks of documented conservative therapy "
    "before advanced imaging. [POL-IMG-004 section 2]"
)

# Clients are cached per provider so repeated calls reuse one connection pool.
_CLIENTS: dict[tuple[str, str, str], Any] = {}


class HashingEmbeddings(Embeddings):
    """Offline, deterministic embeddings for tests and demos.

    Words are hashed into buckets and the vector is length-normalised, so
    texts sharing vocabulary land near each other. Good enough to exercise
    retrieval without touching Azure; never used when provider is 'azure'.
    """

    def __init__(self, size: int = 256) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        import hashlib
        import math
        import re

        vector = [0.0] * self.size
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.size] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _token_provider(scope: str) -> Callable[[], str]:
    """Return a callable that yields a fresh Entra ID bearer token."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    return get_bearer_token_provider(DefaultAzureCredential(), scope)


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return the chat model (the writer)."""
    settings = settings or get_settings()
    key = ("chat", settings.llm_provider, settings.azure_openai_chat_deployment)

    if key not in _CLIENTS:
        if settings.llm_provider == "mock":
            _CLIENTS[key] = GenericFakeChatModel(
                messages=itertools.cycle([AIMessage(content=MOCK_ANSWER)])
            )
        else:
            from langchain_openai import ChatOpenAI

            _CLIENTS[key] = ChatOpenAI(
                model=settings.azure_openai_chat_deployment,
                base_url=settings.azure_openai_base_url,
                api_key=_token_provider(settings.azure_token_scope),
                temperature=0,
            )
    return _CLIENTS[key]


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the embeddings model (the librarian)."""
    settings = settings or get_settings()
    key = (
        "embeddings",
        settings.llm_provider,
        settings.azure_openai_embeddings_deployment,
    )

    if key not in _CLIENTS:
        if settings.llm_provider == "mock":
            _CLIENTS[key] = HashingEmbeddings(size=256)
        else:
            from langchain_openai import OpenAIEmbeddings

            _CLIENTS[key] = OpenAIEmbeddings(
                model=settings.azure_openai_embeddings_deployment,
                base_url=settings.azure_openai_base_url,
                api_key=_token_provider(settings.azure_token_scope),
                tiktoken_model_name=settings.embedding_model_name,
            )
    return _CLIENTS[key]
