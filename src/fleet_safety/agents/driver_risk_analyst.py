"""
Agent 2 — Driver Risk Analyst.

Consumes one driver's history of Agent 1 outputs and asks: is this driver
becoming risky? Unlike Agent 1, most of the answer here is arithmetic
(frequency, weighted severity, a before/after trend comparison) — so this
agent computes that deterministically (stats/driver_risk_stats.py) and
only calls an LLM to turn the computed numbers into grounded explanations.
See driver_risk_stats.py's module docstring for why that split exists.
"""

from __future__ import annotations

import json
import logging

from ..llm.base import LLMClient, LLMError
from ..prompts.driver_risk_analyst import SYSTEM_PROMPT
from ..schemas import (
    CategoryCount,
    DriverRiskInput,
    DriverRiskOutput,
    RecurringPattern,
    RiskLevel,
    TrendDirection,
)
from ..stats.driver_risk_stats import DriverRiskStats, compute_stats

logger = logging.getLogger(__name__)

MAX_NARRATIVE_ATTEMPTS = 2


class DriverRiskAnalyst:
    """Agent 2. Stateless and reusable, same shape as Agent 1: construct
    once with an LLMClient (used only for the narrative layer), call
    analyze() per driver."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze(self, driver_input: DriverRiskInput | dict) -> DriverRiskOutput:
        if isinstance(driver_input, dict):
            driver_input = DriverRiskInput(**driver_input)

        stats = compute_stats(driver_input.incidents, driver_input.time_window_days)
        narrative = self._get_narrative(stats)

        return self._assemble(driver_input, stats, narrative)

    # ------------------------------------------------------------------

    def _get_narrative(self, stats: DriverRiskStats) -> dict:
        """Calls the LLM for explanations only. On any failure (LLM error,
        bad JSON, or a response that doesn't match the patterns it was
        given), falls back to a plain deterministic template rather than
        failing the whole agent — the numeric analysis is already correct
        and complete without this step; the narrative is a nice-to-have,
        not a dependency the agent should be fragile to."""
        if not stats.recurring and stats.total_incidents == 0:
            return {"recurring_patterns": [], "recommended_focus_areas": []}

        payload = {
            "total_incidents": stats.total_incidents,
            "incident_breakdown": [{"category": c, "count": n} for c, n in stats.breakdown],
            "risk_score": stats.risk_score,
            "risk_level": stats.risk_level.value,
            "trend": stats.trend.value,
            "primary_concern": stats.primary_concern,
            "recurring_patterns": [
                {"pattern": g.pattern, "occurrences": g.occurrences, "trend": g.trend.value}
                for g in stats.recurring
            ],
        }
        user_message = json.dumps(payload)

        for attempt in range(1, MAX_NARRATIVE_ATTEMPTS + 1):
            try:
                raw = self.llm_client.complete(SYSTEM_PROMPT, user_message)
                parsed = self._parse_json(raw)
                self._validate_narrative(parsed, stats)
                return parsed
            except (LLMError, json.JSONDecodeError, ValueError) as e:
                logger.warning("Narrative generation failed on attempt %d/%d: %s", attempt, MAX_NARRATIVE_ATTEMPTS, e)

        return self._fallback_narrative(stats)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    @staticmethod
    def _validate_narrative(parsed: dict, stats: DriverRiskStats) -> None:
        """The narrative LLM is only allowed to add text — it must not
        drop, rename, or invent a recurring pattern. Enforced in code, not
        just asked for in the prompt."""
        expected = {g.pattern for g in stats.recurring}
        got = {p.get("pattern") for p in parsed.get("recurring_patterns", [])}
        if expected != got:
            raise ValueError(f"narrative patterns {got} did not match expected {expected}")

    @staticmethod
    def _fallback_narrative(stats: DriverRiskStats) -> dict:
        patterns = [
            {
                "pattern": g.pattern,
                "explanation": f"{g.pattern} occurred {g.occurrences} times in this window "
                f"({g.trend.value.lower().replace('_', ' ')}).",
            }
            for g in stats.recurring
        ]
        focus = [stats.primary_concern] if stats.primary_concern != "None" else []
        for g in stats.recurring[:2]:
            if g.pattern not in focus:
                focus.append(g.pattern)
        return {"recurring_patterns": patterns, "recommended_focus_areas": focus[:4] or ["General monitoring"]}

    # ------------------------------------------------------------------

    @staticmethod
    def _assemble(driver_input: DriverRiskInput, stats: DriverRiskStats, narrative: dict) -> DriverRiskOutput:
        explanations = {p["pattern"]: p.get("explanation", "") for p in narrative.get("recurring_patterns", [])}

        recurring_patterns = [
            RecurringPattern(
                pattern=g.pattern,
                occurrences=g.occurrences,
                trend=g.trend,
                explanation=explanations.get(g.pattern, f"{g.pattern} occurred {g.occurrences} times."),
                example_incident_ids=g.incident_ids[:5],
            )
            for g in stats.recurring
        ]

        limitations = []
        if stats.total_incidents == 0:
            limitations.append("No incident history was provided for this driver in the given window.")
        if stats.undated_count:
            limitations.append(
                f"{stats.undated_count} incident(s) had no timestamp; they were counted toward the risk "
                f"score but excluded from trend analysis."
            )
        if stats.trend == TrendDirection.INSUFFICIENT_DATA and stats.total_incidents > 0:
            limitations.append(
                "Fewer than 4 dated incidents were available, so an increasing/decreasing trend "
                "could not be reliably determined."
            )
        low_conf_sources = sum(1 for i in driver_input.incidents if i.confidence < 55)
        if low_conf_sources:
            limitations.append(
                f"{low_conf_sources} of the underlying incident analyses had confidence below 55; "
                f"this driver risk assessment inherits that uncertainty."
            )

        requires_attention = stats.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and (
            stats.trend == TrendDirection.INCREASING or stats.risk_level == RiskLevel.CRITICAL
        )

        return DriverRiskOutput(
            driver_id=driver_input.driver_id,
            time_window_days=driver_input.time_window_days,
            total_incidents=stats.total_incidents,
            incident_breakdown=[CategoryCount(category=c, count=n) for c, n in stats.breakdown],
            risk_score=stats.risk_score,
            risk_level=stats.risk_level,
            trend=stats.trend,
            primary_concern=stats.primary_concern,
            recurring_patterns=recurring_patterns,
            evidence=stats.evidence,
            confidence=stats.confidence,
            recommended_focus_areas=narrative.get("recommended_focus_areas", []),
            requires_immediate_attention=requires_attention,
            limitations=limitations or ["None identified."],
        )
