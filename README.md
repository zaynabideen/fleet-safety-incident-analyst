# Fleet Safety Intelligence — Agents 1 & 2

Two agents of a multi-agent fleet safety intelligence platform.

- **Agent 1 — Fleet Safety Incident Analyst**: turns one raw driving
  incident (whatever telemetry, environment, and visual data happens to be
  available) into a structured, evidence-based safety assessment — never
  a bare "harsh braking detected" alert.
- **Agent 2 — Driver Risk Analyst**: consumes one driver's history of
  Agent 1 outputs over a time window and answers "is this driver becoming
  risky?" — a risk score, trend, and recurring behaviour patterns.

Full design rationale, schema decisions, and the path to Agent 3 (Fleet
Risk Analyst) are in [ARCHITECTURE.md](./ARCHITECTURE.md).

## Quickstart

```bash
pip install -r requirements.txt

# Run the full test suite (offline, free — no API key needed)
pytest tests/ -v

# Agent 1: run all 7 demo scenarios
python scripts/run_scenarios.py

# Agent 2: run all 5 driver-history demo scenarios
python scripts/run_driver_scenarios.py

# Full pipeline demo: raw incidents -> Agent 1 -> Agent 2, for one driver
python scripts/analyze_driver.py scripts/my_driver.example.json

# Any of the above with --live uses the real Anthropic API instead of the
# offline mock (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
python scripts/run_scenarios.py --live
```

## Testing your own scenario

**One incident (Agent 1 only):**
```bash
cp scripts/my_incident.example.json my_incident.json
# edit my_incident.json
python scripts/analyze_incident.py my_incident.json
```

**A driver's history (full Agent 1 -> Agent 2 pipeline):**
```bash
cp scripts/my_driver.example.json my_driver.json
# edit my_driver.json: driver_id, time_window_days, and a list of raw incidents
python scripts/analyze_driver.py my_driver.json
```

## Using the agents in code

```python
from fleet_safety.agents.incident_analyst import FleetSafetyIncidentAnalyst
from fleet_safety.agents.driver_risk_analyst import DriverRiskAnalyst
from fleet_safety.llm.anthropic_client import AnthropicLLMClient
# offline alternative: from fleet_safety.llm.mock_client import MockLLMClient, DriverRiskMockLLMClient

incident_agent = FleetSafetyIncidentAnalyst(AnthropicLLMClient())
driver_agent = DriverRiskAnalyst(AnthropicLLMClient())

raw_incidents = [ ... ]  # this driver's incidents over the window, IncidentInput-shaped dicts
analyzed = [incident_agent.analyze(raw) for raw in raw_incidents]

driver_risk = driver_agent.analyze({
    "driver_id": "102",
    "time_window_days": 30,
    "incidents": [a.model_dump() for a in analyzed],
})

print(driver_risk.risk_level, driver_risk.trend, driver_risk.primary_concern)
```

## Why two LLM backends, and why Agent 2's is different

`AnthropicLLMClient` is the real thing, shared by both agents. `MockLLMClient`
(Agent 1) and `DriverRiskMockLLMClient` (Agent 2) are deterministic,
zero-cost stand-ins used so the test suite and demos run with no API key
and no network dependency.

For Agent 1, the mock is a rough approximation of real reasoning — it's
useful for testing the pipeline, not a claim of matching an LLM's
judgment. Agent 2 is different: its numbers (risk score, trend, incident
counts) are computed deterministically in plain Python
(`stats/driver_risk_stats.py`) regardless of which LLM backend is used —
the LLM only ever writes the explanation sentences for numbers it's
already been given. See ARCHITECTURE.md section 6.1 for why that split
exists. Practically: Agent 2 gives you the *same* score/trend/level
whether you run it with `--live` or not; only the prose differs.

Swap backends by changing one constructor call; nothing else in the
codebase cares which one is in use.

## Project layout

```
src/fleet_safety/
  schemas.py                  IncidentInput/Output, DriverRiskInput/Output (Pydantic) — the stable contracts
  exceptions.py
  llm/
    base.py                    LLMClient interface
    anthropic_client.py         production backend (Anthropic Messages API), used by both agents
    mock_client.py               MockLLMClient (Agent 1) + DriverRiskMockLLMClient (Agent 2)
  prompts/
    incident_analyst.py         Agent 1 system prompt
    driver_risk_analyst.py       Agent 2 narrative-layer system prompt
  stats/
    driver_risk_stats.py         Agent 2's deterministic scoring/trend engine (no LLM involved)
  agents/
    incident_analyst.py          Agent 1 orchestration
    driver_risk_analyst.py       Agent 2 orchestration: stats engine -> narrative LLM call -> assemble
tests/
  scenarios.py / test_incident_analyst.py            Agent 1: 7 scenarios + mechanics tests
  driver_scenarios.py / test_driver_risk_analyst.py   Agent 2: 5 scenarios + mechanics tests
  test_pipeline_integration.py                         Agent 1 -> Agent 2 handoff, end to end
scripts/
  run_scenarios.py / analyze_incident.py               Agent 1: run demo scenarios / your own incident
  run_driver_scenarios.py                              Agent 2: run demo driver histories
  analyze_driver.py                                    Full pipeline: your raw incidents -> Agent 1 -> Agent 2
```

## Status

Agents 1 and 2 built and tested, per the project's build order (one agent
at a time, connect and stabilize before adding the next). Not built yet:
Fleet Risk Analyst, Coaching Agent, Action Agent, orchestrator, UI, auth,
persistence.
