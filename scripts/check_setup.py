"""Verify that configuration and model access work before building anything else.

Run:  python scripts/check_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.config import get_settings  # noqa: E402
from clinical_rag.llm import get_chat_model, get_embeddings  # noqa: E402


def main() -> int:
    settings = get_settings()

    print("Configuration")
    print(f"  provider            : {settings.llm_provider}")
    print(f"  endpoint            : {settings.azure_openai_base_url or '(mock)'}")
    print(f"  chat deployment     : {settings.azure_openai_chat_deployment}")
    print(f"  embeddings deployment: {settings.azure_openai_embeddings_deployment}")
    print(f"  token scope         : {settings.azure_token_scope}")
    print()

    print("Chat model")
    try:
        reply = get_chat_model().invoke("Reply with the single word: OK")
        print(f"  response: {reply.content!r}")
    except Exception as exc:  # noqa: BLE001 - surface the real error to the user
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1

    print()
    print("Embeddings model")
    try:
        vector = get_embeddings().embed_query("lower back pain MRI coverage")
        preview = ", ".join(f"{v:.4f}" for v in vector[:4])
        print(f"  dimensions: {len(vector)}")
        print(f"  first values: [{preview}, ...]")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1

    print()
    print("Setup OK — no API keys were used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
