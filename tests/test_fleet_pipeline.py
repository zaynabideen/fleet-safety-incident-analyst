"""
Tests for the Fleet Safety Agent 1 -> Agent 2 pipeline built on top of the
generic Orchestrator (orchestration/fleet_pipeline.py). Uses the same
MockLLMClient / DriverRiskMockLLMClient as every other offline test in this
project — zero network calls, zero cost, deterministic.

This is not a re-test of Agent 1 or Agent 2's own reasoning (that's
test_incident_analyst.py / test_driver_risk_analyst.py), and not a re-test
of the raw agent-to-agent handoff (that's test_pipeline_integration.py). It
proves the Orchestrator wraps that same handoff correctly: every incident
attempted, every attempt recorded, and a failure anywhere in Agent 1
stopping the pipeline before Agent 2 ever runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_safety.agents.driver_risk_analyst import DriverRiskAnalyst
from fleet_safety.agents.incident_analyst import FleetSafetyIncidentAnalyst
from fleet_safety.llm.mock_client import DriverRiskMockLLMClient, MockLLMClient
from fleet_safety.orchestration.fleet_pipeline import FleetPipelineInput, build_fleet_safety_pipeline
from fleet_safety.orchestration.types import AgentStatus
from fleet_safety.schemas import DriverRiskOutput, IncidentOutput

# Same escalating-driver story as test_pipeline_integration.py, so the
# expected trend/pattern behaviour here is already known-good.
RAW_INCIDENTS_FOR_DRIVER_102 = [
    {
        "incident_id": "INC-P1", "event_type": "harsh_braking", "vehicle_speed": 35,
        "following_distance": 2.2, "road_condition": "dry", "location_type": "residential street",
        "description": "Minor harsh braking event.", "timestamp": "2026-08-01T08:00:00",
    },
    {
        "incident_id": "INC-P2", "event_type": "harsh_braking", "vehicle_speed": 40,
        "following_distance": 1.3, "road_condition": "dry", "location_type": "highway",
        "description": "Harsh braking during highway driving.", "timestamp": "2026-08-06T08:00:00",
    },
    {
        "incident_id": "INC-P3", "event_type": "harsh_braking", "vehicle_speed": 42,
        "following_distance": 1.1, "road_condition": "wet", "location_type": "highway",
        "description": "Harsh braking during highway driving.", "timestamp": "2026-08-14T08:00:00",
    },
    {
        "incident_id": "INC-P4", "event_type": "harsh_braking", "vehicle_speed": 48,
        "following_distance": 0.9, "road_condition": "wet", "location_type": "highway",
        "description": "Harsh braking during highway driving.", "timestamp": "2026-08-21T08:00:00",
    },
    {
        "incident_id": "INC-P5", "event_type": "harsh_braking", "vehicle_speed": 50,
        "following_distance": 0.8, "road_condition": "dry", "location_type": "highway",
        "description": "Harsh braking during highway driving.", "timestamp": "2026-08-25T08:00:00",
    },
    {
        "incident_id": "INC-P6", "event_type": "harsh_braking", "vehicle_speed": 45,
        "following_distance": 0.7, "road_condition": "dry", "location_type": "highway",
        "description": "Harsh braking during highway driving.", "timestamp": "2026-08-29T08:00:00",
    },
]

# A blank incident_id fails IncidentInput's own field_validator with a
# pydantic ValidationError, raised before FleetSafetyIncidentAnalyst's own
# try/except even begins — exactly the kind of "expected domain error"
# known_errors exists to catch.
BAD_INCIDENT = {"incident_id": "", "event_type": "harsh_braking"}


def _build_pipeline():
    incident_agent = FleetSafetyIncidentAnalyst(MockLLMClient())
    driver_agent = DriverRiskAnalyst(DriverRiskMockLLMClient())
    return build_fleet_safety_pipeline(incident_agent, driver_agent)


def _incident_trace(result):
    return [r for r in result.trace if r.agent_name == "incident_analyst"]


def _driver_risk_trace(result):
    return [r for r in result.trace if r.agent_name == "driver_risk_analyst"]


# ---------------------------------------------------------------------------
# Successful pipeline
# ---------------------------------------------------------------------------

def test_agent1_to_agent2_successful_pipeline():
    orchestrator = _build_pipeline()
    pipeline_input = FleetPipelineInput(
        driver_id="102", time_window_days=30, raw_incidents=RAW_INCIDENTS_FOR_DRIVER_102,
    )

    result = orchestrator.run(pipeline_input)

    assert result.status == AgentStatus.SUCCESS
    assert isinstance(result.result, DriverRiskOutput)
    assert result.result.driver_id == "102"
    assert result.result.total_incidents == 6


def test_multiple_incidents_all_processed():
    orchestrator = _build_pipeline()
    pipeline_input = FleetPipelineInput(
        driver_id="102", time_window_days=30, raw_incidents=RAW_INCIDENTS_FOR_DRIVER_102,
    )

    result = orchestrator.run(pipeline_input)
    incident_trace = _incident_trace(result)

    assert len(incident_trace) == len(RAW_INCIDENTS_FOR_DRIVER_102)
    assert all(r.status == AgentStatus.SUCCESS for r in incident_trace)
    assert [r.item_ref for r in incident_trace] == [i["incident_id"] for i in RAW_INCIDENTS_FOR_DRIVER_102]


def test_agent1_output_correctly_becomes_agent2_input():
    orchestrator = _build_pipeline()
    pipeline_input = FleetPipelineInput(
        driver_id="102", time_window_days=30, raw_incidents=RAW_INCIDENTS_FOR_DRIVER_102,
    )

    result = orchestrator.run(pipeline_input)

    incident_outputs = [r.output for r in _incident_trace(result)]
    assert all(isinstance(o, IncidentOutput) for o in incident_outputs)
    assert all(o.timestamp for o in incident_outputs)

    driver_risk_trace = _driver_risk_trace(result)
    assert len(driver_risk_trace) == 1
    assert driver_risk_trace[0].status == AgentStatus.SUCCESS
    # the same six incidents Agent 1 produced are what Agent 2 saw
    assert result.result.total_incidents == len(incident_outputs)


def test_successful_final_driver_risk_output_reflects_escalating_pattern():
    orchestrator = _build_pipeline()
    pipeline_input = FleetPipelineInput(
        driver_id="102", time_window_days=30, raw_incidents=RAW_INCIDENTS_FOR_DRIVER_102,
    )

    result = orchestrator.run(pipeline_input)

    # Following distance tightened from 2.2s to 0.7s over the month — same
    # assertion test_pipeline_integration.py makes at the raw-agent level;
    # here it's proven through the orchestrator instead.
    assert result.result.trend.value == "INCREASING"
    assert result.result.risk_level.value in ("MEDIUM", "HIGH", "CRITICAL")
    pattern_names = [p.pattern for p in result.result.recurring_patterns]
    assert any("following distance" in p.lower() for p in pattern_names)


# ---------------------------------------------------------------------------
# Deliberate Agent 1 failure
# ---------------------------------------------------------------------------

def test_deliberate_agent1_failure_halts_before_agent2():
    orchestrator = _build_pipeline()
    pipeline_input = FleetPipelineInput(
        driver_id="102", time_window_days=30,
        raw_incidents=[BAD_INCIDENT, *RAW_INCIDENTS_FOR_DRIVER_102],
    )

    result = orchestrator.run(pipeline_input)

    assert result.status == AgentStatus.FAILED
    assert result.failed_at == "incident_analyst"
    assert result.result is None


def test_failed_incident_appears_in_trace():
    orchestrator = _build_pipeline()
    pipeline_input = FleetPipelineInput(
        driver_id="102", time_window_days=30,
        raw_incidents=[BAD_INCIDENT, *RAW_INCIDENTS_FOR_DRIVER_102],
    )

    result = orchestrator.run(pipeline_input)
    incident_trace = _incident_trace(result)

    failed_entries = [r for r in incident_trace if r.status == AgentStatus.FAILED]
    assert len(failed_entries) == 1
    assert failed_entries[0].error_type == "ValidationError"
    assert failed_entries[0].error  # a real message, not blank


def test_agent2_does_not_execute_after_agent1_failure():
    orchestrator = _build_pipeline()
    pipeline_input = FleetPipelineInput(
        driver_id="102", time_window_days=30,
        raw_incidents=[BAD_INCIDENT, *RAW_INCIDENTS_FOR_DRIVER_102],
    )

    result = orchestrator.run(pipeline_input)

    assert _driver_risk_trace(result) == []


def test_no_silent_dropping_every_incident_recorded_even_with_a_failure():
    orchestrator = _build_pipeline()
    bad_incidents = [BAD_INCIDENT, *RAW_INCIDENTS_FOR_DRIVER_102]
    pipeline_input = FleetPipelineInput(driver_id="102", time_window_days=30, raw_incidents=bad_incidents)

    result = orchestrator.run(pipeline_input)
    incident_trace = _incident_trace(result)

    # one bad incident + six good ones = seven attempted, every one
    # recorded in the trace — the failure does not cause the rest of the
    # batch to go unattempted or unrecorded.
    assert len(incident_trace) == len(bad_incidents)
    statuses = [r.status for r in incident_trace]
    assert statuses.count(AgentStatus.FAILED) == 1
    assert statuses.count(AgentStatus.SUCCESS) == len(RAW_INCIDENTS_FOR_DRIVER_102)
