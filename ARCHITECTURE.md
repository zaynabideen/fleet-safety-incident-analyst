# Agent 1 — Fleet Safety Incident Analyst — Design Notes

## 1. What this agent does

Takes one structured driving incident (whatever subset of telemetry,
environment, and visual data is available) and produces a structured,
evidence-based safety assessment: what happened, what contributed to it,
whether the driver was responsible, how serious it was, how confident the
assessment is, and what should happen next. It is the first of five planned
agents in a fleet safety intelligence platform (Incident -> Driver Risk ->
Fleet Risk -> Coaching -> Action), so its output schema is the contract the
rest of the system will be built on.

## 2. Architecture

```
IncidentInput (Pydantic)
        |
        v
FleetSafetyIncidentAnalyst.analyze()
        |
        |-- builds: SYSTEM_PROMPT + incident JSON
        |-- calls:  LLMClient.complete(system, user) -> raw text
        |-- parses: strip markdown fences -> json.loads -> IncidentOutput(**parsed)
        |-- on JSON/validation failure: retry with a repair prompt (up to 3 attempts)
        |-- enforces: safety invariants in code (see 2.3)
        v
IncidentOutput (Pydantic) --> consumed by Agent 2 (Driver Risk Analyst)
```

### 2.1 LLM abstraction (`llm/base.py`)

One method: `complete(system_prompt, user_message) -> str`. That's the
entire surface every future agent needs. Two backends implement it:

- `AnthropicLLMClient` — real reasoning via the Messages API. Requires
  `ANTHROPIC_API_KEY`.
- `MockLLMClient` — a deterministic rule engine that walks the same
  12-step investigation the system prompt describes, with zero network
  calls or API cost. It exists for testing the orchestration layer (schema
  validation, retries, invariant enforcement) and for running the demo
  without requiring an API key — **not** as a claim that rule-based logic
  replaces LLM reasoning. Swapping backends is a one-line change in the
  agent's constructor; nothing else in the codebase knows which one is in
  use.

Why this split instead of one client class with an `if mock:` branch: it
keeps the vendor SDK import out of the agent and schema code entirely, so
a future provider swap (or a fine-tuned/local model) is a new file, not a
refactor.

### 2.2 Schema (`schemas.py`)

`IncidentInput`: only `incident_id` is required. Every telemetry,
environment, and visual field is `Optional`, because the whole point of
the project brief is that real incidents arrive with wildly different
amounts of evidence — a text-only event vs. one with full telemetry and
video. `extra = "allow"` so a future field (e.g. a new sensor channel)
doesn't break existing callers.

`IncidentOutput`: deliberately over-typed relative to "just return JSON" —
`severity`, `driver_contribution.level`, `contributing_factors[].impact`,
and `recommended_action.action` are all enums, and `confidence` /
`root_cause.confidence` are `Field(ge=0, le=100)`. This is what makes the
output "machine-readable" in the sense Agent 2 will need: a Driver Risk
Analyst that's counting `severity == HIGH` occurrences across 30 days of
incidents cannot tolerate a model that sometimes writes `"High"` and
sometimes `"HIGH — likely"`. Pydantic validation turns that failure mode
into an exception the orchestration layer retries on, instead of silent
downstream corruption.

### 2.3 Safety invariants enforced in code, not just prompted for

`_enforce_invariants()` in the agent overrides the model on a small,
deliberately narrow set of things:

- `requires_human_review` is forced `True` whenever severity is
  HIGH/CRITICAL or confidence is below 55, even if the model said
  otherwise.
- `incident_id` in the output is forced to match the input (models
  occasionally transcribe IDs wrong).

This matches the brief's principle #9/#10 (no automated disciplinary or
safety-critical decisions) — the system should not depend on the LLM
reliably remembering to set a flag on every call. A prompt instruction is
a strong suggestion; a code-level check is a guarantee.

### 2.4 Retry / repair loop

Up to 3 attempts. On a JSON parse failure or a Pydantic `ValidationError`,
the next attempt's prompt includes the previous bad response and the
specific error, and asks the model to fix it — this is meaningfully better
than a blind retry because it tells the model exactly what was wrong
(e.g. "severity must be one of LOW/MEDIUM/HIGH/CRITICAL, got 'Medium-High'").
After 3 failed attempts, raises `OutputValidationError` rather than ever
returning a best-effort/partial result — a fleet safety system should fail
loudly, not guess.

## 3. LLM interface details

`AnthropicLLMClient` defaults to `claude-sonnet-4-5-20250929`, temperature
0 (deterministic assessments, not creative ones), 2000 max output tokens.
Model, temperature, and max_tokens are all constructor args so this can be
tuned per deployment without touching the agent.

## 4. Evaluation strategy

Two layers, both in `tests/`:

1. **Scenario tests** (`test_scenario`, parametrized over
   `tests/scenarios.py`) — assert the *kind* of conclusion for each of 7
   realistic incidents, not exact string matches (LLM/rule-engine output
   is not going to be byte-identical run to run). E.g. "pedestrian steps
   into the road" must produce `driver_contribution in {MINOR, NONE}`,
   not `SIGNIFICANT`; "only 'harsh braking detected', nothing else" must
   produce `driver_contribution == UNKNOWN` and root cause "insufficient
   evidence", never a fabricated cause.
2. **Mechanics tests** — retry-on-bad-JSON, markdown-fence stripping,
   raising after max attempts on both a broken-output model and a
   network-failing model, and the invariant-enforcement override — using
   hand-written fake `LLMClient`s so these are independent of any real
   model's reasoning quality.

Both layers run against `MockLLMClient` by default (`pytest tests/`, zero
cost). `scripts/run_scenarios.py --live` runs the same 7 scenarios through
the real Anthropic API when `ANTHROPIC_API_KEY` is set, for a qualitative
check of actual reasoning quality against the same expectations.

The scenario set intentionally includes the failure modes the brief calls
out by name: a defensive-braking case that must not auto-blame the driver,
a bare event with no supporting fields that must not invent a cause, a
possible-distraction case with no camera evidence that must stay
`UNKNOWN` rather than asserting distraction, and a case with contradictory
video vs. description that must lower confidence and force human review.

## 5. Known limitations of this first version

- No real visual/video understanding — `visual_observations` is a text
  field a human or upstream vision model would populate; there's no frame
  extraction or VLM call in this repo yet. The schema and prompt are
  written so that wiring one in later is additive, not a rewrite.
- `MockLLMClient`'s rule engine is intentionally simple (keyword and
  threshold based). It's good enough to exercise the pipeline and pass
  the scenario tests, but it is not a benchmark for reasoning quality —
  that's what the `--live` path with a real model is for.
- Single-incident, stateless. No historical driver context yet (the brief
  explicitly says not to require it for Agent 1) — that's Agent 2's job.

## 6. Path to Agent 2 (Driver Risk Analyst)

`IncidentOutput` is the whole handoff. Agent 2 will consume a list of
`IncidentOutput` objects (one driver's history over N days) and read
`severity`, `contributing_factors`, `driver_contribution`, `root_cause`,
and `evidence` off each — no re-parsing of raw incident data, no
re-deriving facts the first agent already established. Concretely:

```python
history: list[IncidentOutput] = [agent1.analyze(i) for i in driver_102_incidents]
driver_risk = driver_risk_agent.analyze(driver_id="102", incident_history=history)
```

Because the schema is Pydantic and stable, Agent 2's own input schema can
just declare `incident_history: list[IncidentOutput]` and get validation
for free. The same `LLMClient` abstraction, retry/repair loop, and
invariant-enforcement pattern built here carry over directly — Agent 2 is
new prompt + new schema + new rule engine for its mock backend, not a new
orchestration mechanism.

Two fields were added specifically to make this handoff useful, not just
possible: `requires_human_review` (so Agent 2 doesn't have to
re-derive when a human already needed to look at something) and
`limitations` (so a low-confidence incident doesn't silently carry the
same statistical weight as a well-evidenced one when Agent 2 computes a
risk score — it can discount or exclude on that basis).
