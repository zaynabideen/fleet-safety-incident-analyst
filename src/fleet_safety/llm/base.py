"""
LLM abstraction. Every agent in the future platform (Driver Risk, Fleet Risk,
Coaching, Action) will need "send a system prompt + a payload, get text
back" — that's it. Keeping that surface tiny and provider-agnostic here
means:

  - Agent code (agents/incident_analyst.py) never imports a vendor SDK.
  - Swapping Anthropic for another provider, or a fine-tuned/local model
    later, touches one new class, not every agent.
  - Tests run against MockLLMClient with zero network calls and zero cost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """Raised when the LLM backend fails to produce a usable response
    (network/API error, or — after retries — unparseable output)."""


class LLMClient(ABC):
    """Minimal provider-agnostic interface every backend implements."""

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        """
        Send a system prompt and a user message, return the raw text
        response. Implementations should use temperature=0 (or the closest
        equivalent) — this is a safety-analysis agent, not a creative one;
        determinism matters more than variety.

        Must raise LLMError on failure rather than returning an empty or
        partial string, so callers can distinguish "the model said X" from
        "the call failed."
        """
        raise NotImplementedError
