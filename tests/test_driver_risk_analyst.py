"""
Tests for Agent 2, run against DriverRiskMockLLMClient (free, offline —
same pattern as Agent 1's tests). Two layers again: scenario tests against
driver_scenarios.py, and mechanics tests for the narrative fallback path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_safety.agents.driver_risk_analyst import DriverRiskAnalyst
from fleet_safety.llm.mock_client import DriverRiskMockLLMClient
from fleet_safety.schemas import DriverRiskOutput

from driver_scenarios import DRIVER_SCENARIOS


@pytest.fixture
def agent():
    return DriverRiskAnalyst(DriverRiskMockLLMClient())


@pytest.mark.parametrize("scenario", DRIVER_SCENARIOS, ids=[s["name"] for s in DRIVER_SCENARIOS])
def test_driver_scenario(agent, scenario):
    result = agent.analyze(scenario["input"])
    assert isinstance(result, DriverRiskOutput)
    exp = scenario["expect"]

    if "risk_level_in" in exp:
        assert result.risk_level.value in exp["risk_level_in"], (
            f"{scenario['name']}: risk_level={result.risk_level.value}, expected one of {exp['risk_level_in']}"
        )
    if "trend_in" in exp:
        assert result.trend.value in exp["trend_in"], (
            f"{scenario['name']}: trend={result.trend.value}, expected one of {exp['trend_in']}"
        )
    if "requires_immediate_attention" in exp:
        assert result.requires_immediate_attention is exp["requires_immediate_attention"]
    if "primary_concern_in" in exp:
        assert result.primary_concern in exp["primary_concern_in"]
    if "total_incidents" in exp:
        assert result.total_incidents == exp["total_incidents"]
    if "has_recurring_pattern" in exp:
        patterns = [p.pattern for p in result.recurring_patterns]
        assert exp["has_recurring_pattern"] in patterns

    # Universal invariants:
    assert result.driver_id == scenario["input"]["driver_id"]
    assert 0 <= result.risk_score <= 100
    assert 0 <= result.confidence <= 100
    if result.risk_level.value == "CRITICAL":
        assert result.requires_immediate_attention is True


def test_defensive_driving_scores_lower_than_at_fault_driving():
    """The core success criterion for Agent 2, mirroring Agent 1's: a
    driver with several HIGH-severity events they didn't cause must score
    meaningfully lower than a driver with the same severity distribution
    where they were the significant contributor."""
    from driver_scenarios import _incident

    agent = DriverRiskAnalyst(DriverRiskMockLLMClient())

    not_at_fault = agent.analyze({
        "driver_id": "D-A",
        "time_window_days": 30,
        "incidents": [
            _incident(f"A-{i}", "harsh_braking", "HIGH", "NONE", "Sudden external hazard", f"2026-08-{i+1:02d}T09:00:00")
            for i in range(4)
        ],
    })
    at_fault = agent.analyze({
        "driver_id": "D-B",
        "time_window_days": 30,
        "incidents": [
            _incident(f"B-{i}", "harsh_braking", "HIGH", "SIGNIFICANT", "Short following distance", f"2026-08-{i+1:02d}T09:00:00")
            for i in range(4)
        ],
    })
    assert not_at_fault.risk_score < at_fault.risk_score


def test_recurring_pattern_explanation_never_invents_a_pattern():
    """The narrative layer must only explain patterns the deterministic
    stats engine actually found — never add or drop one."""
    from driver_scenarios import _incident

    agent = DriverRiskAnalyst(DriverRiskMockLLMClient())
    result = agent.analyze({
        "driver_id": "D-C",
        "time_window_days": 30,
        "incidents": [
            _incident("C-1", "harsh_braking", "MEDIUM", "MODERATE", "Short following distance", "2026-08-01T09:00:00"),
            _incident("C-2", "harsh_braking", "MEDIUM", "MODERATE", "Short following distance", "2026-08-15T09:00:00"),
        ],
    })
    patterns = {p.pattern for p in result.recurring_patterns}
    assert patterns == {"Short following distance"}
    for p in result.recurring_patterns:
        assert p.explanation  # narrative was actually filled in, not left blank


# ---------------------------------------------------------------------
# Mechanics: narrative fallback when the LLM disagrees with the given facts
# ---------------------------------------------------------------------

class _HallucinatingNarrativeClient:
    """Simulates a model that invents a pattern it wasn't given — the
    agent's _validate_narrative must catch this and fall back to the
    deterministic template instead of trusting it."""

    def complete(self, system_prompt: str, user_message: str) -> str:
        import json
        return json.dumps({
            "recurring_patterns": [{"pattern": "Something I made up", "explanation": "..."}],
            "recommended_focus_areas": ["Made up focus area"],
        })


def test_falls_back_to_deterministic_narrative_on_hallucinated_pattern():
    from driver_scenarios import _incident

    agent = DriverRiskAnalyst(_HallucinatingNarrativeClient())
    result = agent.analyze({
        "driver_id": "D-D",
        "time_window_days": 30,
        "incidents": [
            _incident("D-1", "harsh_braking", "MEDIUM", "MODERATE", "Short following distance", "2026-08-01T09:00:00"),
            _incident("D-2", "harsh_braking", "MEDIUM", "MODERATE", "Short following distance", "2026-08-15T09:00:00"),
        ],
    })
    patterns = {p.pattern for p in result.recurring_patterns}
    assert patterns == {"Short following distance"}  # not the hallucinated one
