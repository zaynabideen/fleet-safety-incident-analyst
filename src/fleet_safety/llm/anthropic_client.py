"""
Production LLM backend: Anthropic's Messages API.

Requires the `anthropic` package (`pip install anthropic`) and an
ANTHROPIC_API_KEY in the environment. Not needed for tests or for the
offline demo — see mock_client.py for the zero-cost path.
"""

from __future__ import annotations

import os

from .base import LLMClient, LLMError

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


class AnthropicLLMClient(LLMClient):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ):
        try:
            import anthropic  # local import: don't force the dependency on
            # anyone only using MockLLMClient
        except ImportError as e:
            raise LLMError(
                "The 'anthropic' package is required for AnthropicLLMClient. "
                "Install it with: pip install anthropic"
            ) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Export it or pass api_key= "
                "explicitly. (Use MockLLMClient if you just want to run "
                "the agent offline.)"
            )

        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as e:  # anthropic raises several distinct
            # exception types (APIConnectionError, RateLimitError,
            # APIStatusError, ...) — the agent only needs to know the
            # call failed and why.
            raise LLMError(f"Anthropic API call failed: {e}") from e

        text_parts = [block.text for block in response.content if block.type == "text"]
        text = "".join(text_parts).strip()
        if not text:
            raise LLMError("Anthropic API returned an empty response.")
        return text
