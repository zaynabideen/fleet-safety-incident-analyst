"""
End-to-end: raw incidents -> Agent 1 -> Agent 2, exactly the handoff
described in ARCHITECTURE.md section 6. This is the test that actually
proves the two agents connect, not just that each works in isolation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_safety.agents.driver_risk_analyst import DriverRiskAnalyst
from fleet_safety.agents.incident_analyst import FleetSafetyIncidentAnalyst
from fleet_safety.llm.mock_client import DriverRiskMockLLMClient, MockLLMClient
from fleet_safety.schemas import DriverRiskInput, DriverRiskOutput

# Six raw incidents for one driver, escalating in severity and following
# too closely more often as the month goes on — same story as the
# "escalating_high_risk_driver" scenario, but starting from RAW incident
# data (what Agent 1 actually receives), not pre-built IncidentOutput.
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


def test_agent1_output_feeds_directly_into_agent2():
    incident_agent = FleetSafetyIncidentAnalyst(MockLLMClient())
    driver_agent = DriverRiskAnalyst(DriverRiskMockLLMClient())

    # Agent 1 runs on each raw incident independently, exactly as it would
    # in production (it has no concept of "driver 102" or history).
    analyzed = [incident_agent.analyze(raw) for raw in RAW_INCIDENTS_FOR_DRIVER_102]
    assert all(a.timestamp for a in analyzed), "Agent 1 must carry timestamps through for Agent 2 to use"

    # The caller (not either agent) is what associates incidents with a
    # driver — that's intentional, see ARCHITECTURE.md section 6.
    driver_result = driver_agent.analyze(DriverRiskInput(
        driver_id="102",
        time_window_days=30,
        incidents=analyzed,
    ))

    assert isinstance(driver_result, DriverRiskOutput)
    assert driver_result.total_incidents == 6
    # Following distance tightened from 2.2s to 0.7s over the month — this
    # should read as a worsening, following-distance-driven pattern.
    assert driver_result.trend.value == "INCREASING"
    assert driver_result.risk_level.value in ("MEDIUM", "HIGH", "CRITICAL")
    pattern_names = [p.pattern for p in driver_result.recurring_patterns]
    assert any("following distance" in p.lower() for p in pattern_names)
