"""Verify access to the Nebius Token Factory model list."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main() -> int:
    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv("NEBIUS_BASE_URL")

    missing = [
        name
        for name, value in (
            ("NEBIUS_API_KEY", api_key),
            ("NEBIUS_BASE_URL", base_url),
        )
        if not value
    ]
    if missing:
        print(f"Missing required environment variable(s): {', '.join(missing)}")
        return 1

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
    except Exception:
        print(f"Nebius connection: failed ({type(exc).__name__})")
        return 1

    print("Nebius connection: successful")
    print("Available model IDs:")
    for model in models.data:
        print(f"- {model.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
