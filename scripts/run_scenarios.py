#!/usr/bin/env python3
"""
Run every scenario in tests/scenarios.py through Agent 1 and print the
structured result. Defaults to MockLLMClient (free, offline). Pass
--live to use AnthropicLLMClient instead (requires ANTHROPIC_API_KEY).

Usage:
    python scripts/run_scenarios.py
    python scripts/run_scenarios.py --live
    python scripts/run_scenarios.py --live --model claude-opus-4-1-20250805
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from fleet_safety.agents.incident_analyst import FleetSafetyIncidentAnalyst
from fleet_safety.exceptions import OutputValidationError
from scenarios import SCENARIOS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use the real Anthropic API instead of the offline mock.")
    parser.add_argument("--model", default=None, help="Override the model id when --live is set.")
    args = parser.parse_args()

    if args.live:
        from fleet_safety.llm.anthropic_client import AnthropicLLMClient
        kwargs = {"model": args.model} if args.model else {}
        llm = AnthropicLLMClient(**kwargs)
        print(f"[mode] LIVE — {llm.model}\n")
    else:
        from fleet_safety.llm.mock_client import MockLLMClient
        llm = MockLLMClient()
        print("[mode] MOCK (offline, deterministic rule engine — set --live for real LLM reasoning)\n")

    agent = FleetSafetyIncidentAnalyst(llm)

    passed, failed = 0, 0
    for scenario in SCENARIOS:
        print("=" * 88)
        print(f"SCENARIO: {scenario['name']}")
        print(f"  {scenario['description']}")
        print("-" * 88)
        try:
            result = agent.analyze(scenario["input"])
            print(json.dumps(result.model_dump(mode="json"), indent=2))
            passed += 1
        except OutputValidationError as e:
            print(f"FAILED: {e}")
            failed += 1
        print()

    print("=" * 88)
    print(f"{passed} succeeded, {failed} failed, out of {len(SCENARIOS)} scenarios.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
