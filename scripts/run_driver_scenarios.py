#!/usr/bin/env python3
"""
Run every scenario in tests/driver_scenarios.py through Agent 2 and print
the structured result. Mirrors scripts/run_scenarios.py for Agent 1.

Usage:
    python scripts/run_driver_scenarios.py
    python scripts/run_driver_scenarios.py --live
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from fleet_safety.agents.driver_risk_analyst import DriverRiskAnalyst
from driver_scenarios import DRIVER_SCENARIOS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use the real Anthropic API for the narrative layer.")
    args = parser.parse_args()

    if args.live:
        from fleet_safety.llm.anthropic_client import AnthropicLLMClient
        llm = AnthropicLLMClient()
        print(f"[mode] LIVE narrative — {llm.model}\n")
    else:
        from fleet_safety.llm.mock_client import DriverRiskMockLLMClient
        llm = DriverRiskMockLLMClient()
        print("[mode] MOCK narrative (offline, template-based — the risk score/trend/counts below are\n"
              " always deterministic Python either way; only the explanation sentences differ)\n")

    agent = DriverRiskAnalyst(llm)

    for scenario in DRIVER_SCENARIOS:
        print("=" * 88)
        print(f"SCENARIO: {scenario['name']}")
        print(f"  {scenario['description']}")
        print("-" * 88)
        result = agent.analyze(scenario["input"])
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        print()


if __name__ == "__main__":
    main()
