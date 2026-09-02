#!/usr/bin/env python3
"""
Full pipeline demo: a list of YOUR raw incidents for one driver -> Agent 1
(each incident analyzed independently) -> Agent 2 (pattern across the
driver's history). This is the Agent 1 -> Agent 2 handoff described in
ARCHITECTURE.md section 6, run end to end on data you provide.

Usage:
    cp scripts/my_driver.example.json my_driver.json
    # edit my_driver.json: driver_id, time_window_days, and a list of raw incidents
    python scripts/analyze_driver.py my_driver.json
    python scripts/analyze_driver.py my_driver.json --live   # real Claude for both agents
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_safety.agents.driver_risk_analyst import DriverRiskAnalyst
from fleet_safety.agents.incident_analyst import FleetSafetyIncidentAnalyst
from fleet_safety.exceptions import OutputValidationError
from fleet_safety.schemas import DriverRiskInput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("driver_file", help="Path to a JSON file: {driver_id, time_window_days, incidents: [...]}")
    parser.add_argument("--live", action="store_true", help="Use the real Anthropic API for both agents.")
    args = parser.parse_args()

    path = Path(args.driver_file)
    if not path.exists():
        print(f"File not found: {path}")
        print("Copy scripts/my_driver.example.json first and edit it.")
        sys.exit(1)

    data = json.loads(path.read_text())
    driver_id = data["driver_id"]
    time_window_days = data.get("time_window_days", 30)
    raw_incidents = data["incidents"]

    if args.live:
        from fleet_safety.llm.anthropic_client import AnthropicLLMClient
        incident_llm = AnthropicLLMClient()
        driver_llm = AnthropicLLMClient()
        print(f"[mode] LIVE — {incident_llm.model}\n")
    else:
        from fleet_safety.llm.mock_client import DriverRiskMockLLMClient, MockLLMClient
        incident_llm = MockLLMClient()
        driver_llm = DriverRiskMockLLMClient()
        print("[mode] MOCK (offline — add --live for real Claude reasoning)\n")

    incident_agent = FleetSafetyIncidentAnalyst(incident_llm)
    driver_agent = DriverRiskAnalyst(driver_llm)

    print("=" * 88)
    print(f"STAGE 1 — Agent 1 analyzing {len(raw_incidents)} raw incident(s) for driver {driver_id}")
    print("=" * 88)
    analyzed = []
    for raw in raw_incidents:
        try:
            result = incident_agent.analyze(raw)
        except OutputValidationError as e:
            print(f"  {raw.get('incident_id', '?')}: FAILED — {e}")
            continue
        analyzed.append(result)
        print(f"  {result.incident_id}: {result.event_type} | severity={result.severity.value} | "
              f"driver_contribution={result.driver_contribution.level.value} | root_cause={result.root_cause.cause}")

    print()
    print("=" * 88)
    print(f"STAGE 2 — Agent 2 analyzing driver {driver_id}'s pattern across {len(analyzed)} analyzed incident(s)")
    print("=" * 88)
    driver_result = driver_agent.analyze(DriverRiskInput(
        driver_id=driver_id,
        time_window_days=time_window_days,
        incidents=analyzed,
    ))
    print(json.dumps(driver_result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
