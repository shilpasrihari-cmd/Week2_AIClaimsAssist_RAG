"""Run a minimal Nebius Token Factory chat-completion test."""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


MODEL_ID = "google/gemma-3-27b-it"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main() -> int:
    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv("NEBIUS_BASE_URL")
    if not api_key or not base_url:
        print("Missing NEBIUS_API_KEY or NEBIUS_BASE_URL")
        return 1

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": "Explain collision coverage in one sentence.",
                }
            ],
        )
        latency = time.perf_counter() - start
    except Exception as exc:
        print(f"Nebius LLM request failed ({type(exc).__name__})")
        return 1

    print(f"Model: {MODEL_ID}")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Response: {response.choices[0].message.content}")
    if response.usage:
        print(f"Token usage: {response.usage}")
    else:
        print("Token usage: unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
