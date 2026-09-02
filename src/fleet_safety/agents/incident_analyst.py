"""
Agent 1 — Fleet Safety Incident Analyst.

Orchestration only. All investigation logic lives in the system prompt (for
a real LLM backend) or the rule engine (for MockLLMClient) — this class's
job is: build the call, get structured output back reliably, and enforce
the handful of safety invariants that must hold no matter what the model
says.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from ..exceptions import OutputValidationError
from ..llm.base import LLMClient, LLMError
from ..prompts.incident_analyst import SYSTEM_PROMPT
from ..schemas import IncidentInput, IncidentOutput

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


class FleetSafetyIncidentAnalyst:
    """Agent 1. Stateless and reusable — construct once with an LLMClient,
    call analyze() per incident."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze(self, incident: IncidentInput | dict) -> IncidentOutput:
        if isinstance(incident, dict):
            incident = IncidentInput(**incident)

        user_message = incident.model_dump_json(exclude_none=True)

        last_error: Exception | None = None
        raw_response = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            prompt = user_message if attempt == 1 else self._repair_prompt(
                user_message, raw_response, last_error
            )
            try:
                raw_response = self.llm_client.complete(SYSTEM_PROMPT, prompt)
            except LLMError as e:
                last_error = e
                logger.warning("LLM call failed on attempt %d/%d: %s", attempt, MAX_ATTEMPTS, e)
                continue

            try:
                parsed = self._parse_json(raw_response)
                output = IncidentOutput(**parsed)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    "Output failed validation on attempt %d/%d: %s", attempt, MAX_ATTEMPTS, e
                )
                continue

            return self._enforce_invariants(output, incident)

        raise OutputValidationError(
            f"Agent failed to produce a valid IncidentOutput after {MAX_ATTEMPTS} attempts "
            f"for incident {incident.incident_id!r}. Last error: {last_error}"
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Models occasionally wrap JSON in ```json fences despite
        instructions not to. Strip that before parsing rather than failing
        the whole attempt over formatting."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    @staticmethod
    def _repair_prompt(original_user_message: str, bad_response: str, error: Exception | None) -> str:
        return (
            f"Your previous response could not be parsed/validated as the required JSON schema.\n"
            f"Error: {error}\n\n"
            f"Previous response:\n{bad_response}\n\n"
            f"Re-analyze the following incident and return ONLY a single valid JSON object matching "
            f"the required schema exactly — no markdown fences, no extra text.\n\n{original_user_message}"
        )

    @staticmethod
    def _enforce_invariants(output: IncidentOutput, incident: IncidentInput) -> IncidentOutput:
        """
        Deterministic safety net on top of whatever the LLM decided. This
        project's core principle is that the system must never let an
        automated judgment call override a human on a high-stakes case —
        so these checks are enforced in code, not left to prompt
        compliance alone.
        """
        notes: list[str] = []

        if output.severity.value in ("HIGH", "CRITICAL") and not output.requires_human_review:
            output.requires_human_review = True
            notes.append(
                f"Human review auto-required: severity was {output.severity.value}."
            )

        if output.confidence < 55 and not output.requires_human_review:
            output.requires_human_review = True
            notes.append(f"Human review auto-required: confidence ({output.confidence}) below threshold.")

        if output.incident_id != incident.incident_id:
            notes.append(
                f"incident_id mismatch corrected: model returned {output.incident_id!r}, "
                f"expected {incident.incident_id!r}."
            )
            output.incident_id = incident.incident_id

        if notes:
            output.limitations = [*output.limitations, *notes]

        return output
