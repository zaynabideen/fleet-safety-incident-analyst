#!/usr/bin/env python3
"""
Run Agent 1 on ONE incident that YOU provide, instead of the built-in demo
scenarios.

Usage:
    # 1. Copy the template and fill in your own incident:
    cp scripts/my_incident.example.json my_incident.json
    # edit my_incident.json in any text editor

    # 2. Run it (offline mock — free, no API key):
    python scripts/analyze_incident.py my_incident.json

    # 3. Run it with the real Claude model instead (needs ANTHROPIC_API_KEY):
    python scripts/analyze_incident.py my_incident.json --live
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_safety.agents.incident_analyst import FleetSafetyIncidentAnalyst
from fleet_safety.exceptions import OutputValidationError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("incident_file", help="Path to a JSON file describing your incident.")
    parser.add_argument("--live", action="store_true", help="Use the real Anthropic API instead of the offline mock.")
    args = parser.parse_args()

    incident_path = Path(args.incident_file)
    if not incident_path.exists():
        print(f"File not found: {incident_path}")
        print("Copy scripts/my_incident.example.json first and edit it.")
        sys.exit(1)

    incident = json.loads(incident_path.read_text())

    if args.live:
        from fleet_safety.llm.anthropic_client import AnthropicLLMClient
        llm = AnthropicLLMClient()
        print(f"[mode] LIVE — {llm.model}\n")
    else:
        from fleet_safety.llm.mock_client import MockLLMClient
        llm = MockLLMClient()
        print("[mode] MOCK (offline rule engine — add --live for real Claude reasoning)\n")

    agent = FleetSafetyIncidentAnalyst(llm)

    try:
        result = agent.analyze(incident)
    except OutputValidationError as e:
        print(f"Agent failed: {e}")
        sys.exit(1)

    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
