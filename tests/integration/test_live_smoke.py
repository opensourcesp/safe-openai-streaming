from __future__ import annotations

import logging
import os

import pytest

from safe_openai import SafeOpenAI

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_live_safe_stream_integrity() -> None:
    """Opt-in live smoke test for a real OpenAI streaming request."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not configured; skipping live integration test")

    client = SafeOpenAI(api_key=api_key)
    try:
        stream = client.chat.completions.create(
            model=os.environ.get("SAFE_OPENAI_SMOKE_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            stream=True,
        )

        full_response = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content

        assert full_response.strip()
        logger.info("Live SafeOpenAI stream smoke test received response: %s", full_response.strip())
    finally:
        client.close()
