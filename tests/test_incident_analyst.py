"""
Tests run entirely against MockLLMClient — no network, no API key, no cost.
Two layers are tested:

  1. Scenario tests: does the agent reach the right *kind* of conclusion
     for each of the realistic incidents in scenarios.py (this is what
     the project brief's success criteria actually asks for).
  2. Mechanics tests: does the orchestration layer (schema validation,
     JSON-fence stripping, retry-on-bad-output, safety-invariant
     enforcement) behave correctly, independent of any LLM's reasoning
     quality.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_safety.agents.incident_analyst import FleetSafetyIncidentAnalyst
from fleet_safety.exceptions import OutputValidationError
from fleet_safety.llm.base import LLMClient, LLMError
from fleet_safety.llm.mock_client import MockLLMClient
from fleet_safety.schemas import IncidentOutput

from scenarios import SCENARIOS


@pytest.fixture
def agent():
    return FleetSafetyIncidentAnalyst(MockLLMClient())


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_scenario(agent, scenario):
    result = agent.analyze(scenario["input"])
    assert isinstance(result, IncidentOutput)
    exp = scenario["expect"]

    if "severity_in" in exp:
        assert result.severity.value in exp["severity_in"], (
            f"{scenario['name']}: severity={result.severity.value}, expected one of {exp['severity_in']}"
        )
    if "driver_contribution_in" in exp:
        assert result.driver_contribution.level.value in exp["driver_contribution_in"], (
            f"{scenario['name']}: driver_contribution={result.driver_contribution.level.value}, "
            f"expected one of {exp['driver_contribution_in']}"
        )
    if "driver_contribution_not_in" in exp:
        assert result.driver_contribution.level.value not in exp["driver_contribution_not_in"]
    if "recommended_action_in" in exp:
        assert result.recommended_action.action.value in exp["recommended_action_in"]
    if "recommended_action_not_in" in exp:
        assert result.recommended_action.action.value not in exp["recommended_action_not_in"]
    if "root_cause_not_unknown" in exp:
        assert result.root_cause.cause.lower() not in ("unknown", "")
    if "root_cause_is_unknown_or_insufficient" in exp:
        assert (
            result.root_cause.cause.lower() == "unknown"
            or "insufficient" in result.root_cause.explanation.lower()
            or "insufficient" in result.root_cause.cause.lower()
        )
    if "confidence_below" in exp:
        assert result.confidence < exp["confidence_below"]
    if "requires_human_review_true" in exp:
        assert result.requires_human_review is True
    if "limitations_mention_distraction_unconfirmed" in exp:
        joined = " ".join(result.limitations).lower()
        assert "distract" in joined

    # Universal invariants, regardless of scenario:
    assert result.incident_id == scenario["input"]["incident_id"]
    if result.severity.value in ("HIGH", "CRITICAL"):
        assert result.requires_human_review is True, "HIGH/CRITICAL must always require human review"
    assert 0 <= result.confidence <= 100
    assert 0 <= result.root_cause.confidence <= 100


def test_different_scenarios_produce_different_outputs(agent):
    """Guards against the agent degenerating into one templated answer for
    every harsh_braking event — the central success criterion."""
    outputs = [agent.analyze(s["input"]) for s in SCENARIOS]
    signatures = {(o.severity.value, o.driver_contribution.level.value, o.recommended_action.action.value) for o in outputs}
    assert len(signatures) > 1


def test_never_invents_data_for_bare_event(agent):
    result = agent.analyze({"incident_id": "INC-BARE", "event_type": "harsh_braking"})
    assert result.observed_facts == [] or all(
        "harsh" not in f.lower() or "detected" not in f.lower() for f in result.observed_facts
    )
    assert result.driver_contribution.level.value == "UNKNOWN"


# ---------------------------------------------------------------------
# Mechanics: retry / repair loop and JSON-fence stripping
# ---------------------------------------------------------------------

class _FlakyThenGoodClient(LLMClient):
    """Returns markdown-fenced JSON on the first call (which the parser
    must strip), then fails validation once, then succeeds — exercises
    both the fence-stripping path and the retry path in one client."""

    def __init__(self):
        self.calls = 0

    def complete(self, system_prompt: str, user_message: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return "```json\n" + MockLLMClient().complete(system_prompt, user_message) + "\n```"
        if self.calls == 2:
            return '{"incident_id": "X"}'  # missing required fields -> ValidationError
        return MockLLMClient().complete(system_prompt, user_message)


def test_strips_markdown_fences_and_retries_on_bad_output():
    client = _FlakyThenGoodClient()
    agent = FleetSafetyIncidentAnalyst(client)
    result = agent.analyze({"incident_id": "INC-001", "event_type": "harsh_braking", "vehicle_speed": 40})
    assert isinstance(result, IncidentOutput)


class _AlwaysBrokenClient(LLMClient):
    def complete(self, system_prompt: str, user_message: str) -> str:
        return "not json at all"


def test_raises_after_max_attempts_on_persistently_bad_output():
    agent = FleetSafetyIncidentAnalyst(_AlwaysBrokenClient())
    with pytest.raises(OutputValidationError):
        agent.analyze({"incident_id": "INC-BAD", "event_type": "harsh_braking"})


class _AlwaysFailingLLM(LLMClient):
    def complete(self, system_prompt: str, user_message: str) -> str:
        raise LLMError("simulated network failure")


def test_raises_after_max_attempts_on_llm_errors():
    agent = FleetSafetyIncidentAnalyst(_AlwaysFailingLLM())
    with pytest.raises(OutputValidationError):
        agent.analyze({"incident_id": "INC-NET", "event_type": "harsh_braking"})


# ---------------------------------------------------------------------
# Mechanics: safety-invariant enforcement (code-level, not prompt-level)
# ---------------------------------------------------------------------

class _OverconfidentHighSeverityClient(LLMClient):
    """Simulates a model that assigns HIGH severity but forgets to flag
    human review — the agent's _enforce_invariants must correct this."""

    def complete(self, system_prompt: str, user_message: str) -> str:
        import json
        return json.dumps({
            "incident_id": "INC-009",
            "incident_summary": "Test",
            "event_type": "harsh_braking",
            "severity": "HIGH",
            "confidence": 90,
            "observed_facts": ["Vehicle speed was 60 mph."],
            "contributing_factors": [{"factor": "Excessive speed", "impact": "HIGH", "evidence": "60 mph."}],
            "driver_contribution": {"level": "SIGNIFICANT", "explanation": "Speed was excessive."},
            "root_cause": {"cause": "Excessive speed", "confidence": 90, "explanation": "Speed was excessive."},
            "evidence": ["Vehicle speed was 60 mph."],
            "recommended_action": {"action": "MANAGER_REVIEW", "reason": "High severity."},
            "requires_human_review": False,  # <- model got this wrong
            "limitations": [],
        })


def test_enforces_human_review_for_high_severity_even_if_model_omits_it():
    agent = FleetSafetyIncidentAnalyst(_OverconfidentHighSeverityClient())
    result = agent.analyze({"incident_id": "INC-009", "event_type": "harsh_braking", "vehicle_speed": 60})
    assert result.requires_human_review is True
    assert any("auto-required" in note for note in result.limitations)
