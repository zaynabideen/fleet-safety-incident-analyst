#!/usr/bin/env python3
"""
Clean demonstration entrypoint for the orchestrated Fleet Safety pipeline:
Agent 1 (Incident Analyst) -> Agent 2 (Driver Risk Analyst), run through
the Orchestrator in src/fleet_safety/orchestration/, instead of the manual
loop scripts/analyze_driver.py builds by hand.

Usage:
    python scripts/run_pipeline.py                        # mock backends, built-in demo data
    python scripts/run_pipeline.py my_driver.json          # your own driver file (see
                                                             # scripts/my_driver.example.json)
    python scripts/run_pipeline.py my_driver.json --live   # real Anthropic API for both agents
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_safety.agents.driver_risk_analyst import DriverRiskAnalyst
from fleet_safety.agents.incident_analyst import FleetSafetyIncidentAnalyst
from fleet_safety.orchestration.fleet_pipeline import FleetPipelineInput, build_fleet_safety_pipeline
from fleet_safety.orchestration.types import AgentStatus, PipelineResult

DEMO_RAW_INCIDENTS = [
    {
        "incident_id": "INC-D1", "event_type": "harsh_braking", "vehicle_speed": 35,
        "following_distance": 2.2, "road_condition": "dry", "location_type": "residential street",
        "description": "Minor harsh braking event.", "timestamp": "2026-08-01T08:00:00",
    },
    {
        "incident_id": "INC-D2", "event_type": "harsh_braking", "vehicle_speed": 42,
        "following_distance": 1.1, "road_condition": "wet", "location_type": "highway",
        "description": "Harsh braking during highway driving, vehicle ahead stopped suddenly.",
        "timestamp": "2026-08-14T08:00:00",
    },
    {
        "incident_id": "INC-D3", "event_type": "harsh_braking", "vehicle_speed": 50,
        "following_distance": 0.7, "road_condition": "dry", "location_type": "highway",
        "description": "Harsh braking during highway driving.", "timestamp": "2026-08-25T08:00:00",
    },
    {
        "incident_id": "INC-D4", "event_type": "harsh_braking", "vehicle_speed": 45,
        "following_distance": 0.6, "road_condition": "dry", "location_type": "highway",
        "description": "Harsh braking during highway driving.", "timestamp": "2026-08-29T08:00:00",
    },
]


def _load_pipeline_input(path: str | None) -> FleetPipelineInput:
    if path is None:
        return FleetPipelineInput(driver_id="DEMO-102", time_window_days=30, raw_incidents=DEMO_RAW_INCIDENTS)
    data = json.loads(Path(path).read_text())
    return FleetPipelineInput(
        driver_id=data["driver_id"],
        time_window_days=data.get("time_window_days", 30),
        raw_incidents=data["incidents"],
    )


def _print_summary(result: PipelineResult) -> None:
    print("=" * 78)
    print(f"PIPELINE STATUS: {result.status.value}")
    print("=" * 78)

    for entry in result.trace:
        ref = f" [{entry.item_ref}]" if entry.item_ref else ""
        duration = f"{entry.duration_ms:.1f}ms" if entry.duration_ms is not None else "n/a"
        if entry.status == AgentStatus.SUCCESS:
            print(f"  OK    {entry.agent_name}{ref}  ({duration})")
        else:
            print(f"  FAIL  {entry.agent_name}{ref}  ({duration}) -- {entry.error_type}: {entry.error}")

    print("-" * 78)

    if result.status == AgentStatus.FAILED:
        print(f"Pipeline halted at stage: {result.failed_at!r}")
        print("No final result -- a downstream stage did not run.")
        return

    driver_risk = result.result
    print(f"Driver:                        {driver_risk.driver_id}")
    print(f"Window:                        {driver_risk.time_window_days} days")
    print(f"Total incidents:               {driver_risk.total_incidents}")
    print(f"Risk score:                    {driver_risk.risk_score} ({driver_risk.risk_level.value})")
    print(f"Trend:                         {driver_risk.trend.value}")
    print(f"Primary concern:               {driver_risk.primary_concern}")
    print(f"Requires immediate attention:  {driver_risk.requires_immediate_attention}")
    if driver_risk.recurring_patterns:
        print("Recurring patterns:")
        for p in driver_risk.recurring_patterns:
            print(f"  - {p.pattern} ({p.occurrences}x, {p.trend.value}): {p.explanation}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "driver_file", nargs="?", default=None,
        help="Path to a driver JSON file (driver_id, time_window_days, incidents). "
        "Omit to run the built-in demo data.",
    )
    parser.add_argument("--live", action="store_true", help="Use the real Anthropic API for both agents.")
    args = parser.parse_args()

    pipeline_input = _load_pipeline_input(args.driver_file)

    if args.live:
        from fleet_safety.llm.anthropic_client import AnthropicLLMClient
        incident_agent = FleetSafetyIncidentAnalyst(AnthropicLLMClient())
        driver_agent = DriverRiskAnalyst(AnthropicLLMClient())
        print("[mode] LIVE -- Anthropic API\n")
    else:
        from fleet_safety.llm.mock_client import DriverRiskMockLLMClient, MockLLMClient
        incident_agent = FleetSafetyIncidentAnalyst(MockLLMClient())
        driver_agent = DriverRiskAnalyst(DriverRiskMockLLMClient())
        print("[mode] MOCK (offline -- add --live for real Claude reasoning)\n")

    orchestrator = build_fleet_safety_pipeline(incident_agent, driver_agent)
    result = orchestrator.run(pipeline_input)

    _print_summary(result)

    sys.exit(0 if result.status == AgentStatus.SUCCESS else 1)


if __name__ == "__main__":
    main()
