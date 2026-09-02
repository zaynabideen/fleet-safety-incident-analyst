# Fleet Safety Incident Analyst (Agent 1)

The first agent in a multi-agent fleet safety intelligence platform. Turns
a raw driving incident (whatever telemetry, environment, and visual data
happens to be available) into a structured, evidence-based safety
assessment — never a bare "harsh braking detected" alert.

Full design rationale, schema decisions, and the path to Agent 2 (Driver
Risk Analyst) are in [ARCHITECTURE.md](./ARCHITECTURE.md).

## Quickstart

```bash
pip install -r requirements.txt

# Run the test suite (offline, free — no API key needed)
pytest tests/ -v

# Run all 7 demo scenarios and print full structured output
python scripts/run_scenarios.py

# Run the same scenarios through the real Anthropic API instead of the
# offline rule-engine mock (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
python scripts/run_scenarios.py --live
```

## Using the agent in code

```python
from fleet_safety.agents.incident_analyst import FleetSafetyIncidentAnalyst
from fleet_safety.llm.anthropic_client import AnthropicLLMClient
# or: from fleet_safety.llm.mock_client import MockLLMClient  (no API key needed)

agent = FleetSafetyIncidentAnalyst(AnthropicLLMClient())

result = agent.analyze({
    "incident_id": "INC-001",
    "event_type": "harsh_braking",
    "vehicle_speed": 42,
    "speed_unit": "mph",
    "following_distance": 1.4,
    "weather": "rain",
    "road_condition": "wet",
    "location_type": "junction",
    "description": "Vehicle ahead stopped suddenly.",
})

print(result.severity, result.driver_contribution.level, result.recommended_action.action)
result.model_dump_json(indent=2)  # the full structured output
```

## Why two LLM backends

`AnthropicLLMClient` is the real thing. `MockLLMClient` is a deterministic,
zero-cost rule engine that implements the same investigation steps as the
system prompt, used so the test suite and demo run with no API key and no
network dependency. See ARCHITECTURE.md section 2.1 for why this isn't
just a stub — it's what lets `test_different_scenarios_produce_different_outputs`
and friends run in CI for free. Swap backends by changing one constructor
call; nothing else in the codebase cares which one is in use.

## Project layout

```
src/fleet_safety/
  schemas.py              IncidentInput / IncidentOutput (Pydantic) — the stable contract
  exceptions.py
  llm/
    base.py                LLMClient interface
    anthropic_client.py     production backend (Anthropic Messages API)
    mock_client.py           offline rule-engine backend, for tests/demo
  prompts/
    incident_analyst.py     system prompt for the real LLM backend
  agents/
    incident_analyst.py      orchestration: prompt -> LLM -> parse -> validate -> retry -> invariants
tests/
  scenarios.py              7 realistic incidents + expected-conclusion assertions
  test_incident_analyst.py  scenario tests + retry/repair/invariant mechanics tests
scripts/
  run_scenarios.py          CLI: run every scenario, print structured output
```

## Status

Agent 1 only, per the project's build order (one agent at a time, stabilize
before adding the next). Not built yet: Driver Risk Analyst, Fleet Risk
Analyst, Coaching Agent, Action Agent, orchestrator, UI, auth, persistence.
