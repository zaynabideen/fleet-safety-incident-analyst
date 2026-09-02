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

## 6. Agent 2 — Driver Risk Analyst

Built. Question it answers: "is this driver becoming risky?" — given one
driver's history of Agent 1 outputs over a time window.

### 6.1 The one design decision worth defending: split the numbers from the words

Agent 1's job (read messy, unstructured evidence about a single event and
form a qualitative judgment) is a genuinely good fit for an LLM. Agent 2's
job is mostly arithmetic: count incidents by type, weight them by severity
and by how much the driver actually contributed, compare the first half of
the window to the second half to call a trend. That's not a judgment call
with room for reasonable disagreement — it's a formula with one correct
answer for a given input.

So `stats/driver_risk_stats.py` computes all of that in plain Python —
`risk_score`, `risk_level`, `trend`, `incident_breakdown`, which
`recurring_patterns` exist and how many times each occurred — with zero
LLM involvement. The LLM (`prompts/driver_risk_analyst.py`) is only ever
handed those already-computed numbers and asked to write grounded
sentences explaining them (`recurring_patterns[].explanation`,
`recommended_focus_areas`). It is explicitly told not to introduce a
number it wasn't given, and the agent enforces that in code
(`_validate_narrative`): if the model's response mentions a pattern that
doesn't match what the stats engine actually found, the agent discards it
and falls back to a deterministic template rather than trusting it. Same
principle as Agent 1's `_enforce_invariants` — a safety-relevant number
should never depend on a language model remembering to stay in its lane.

This also means Agent 2 still works, fully, with zero API calls — the
`DriverRiskMockLLMClient` narrative fallback and the "real" narrative
path differ only in prose quality, never in the score, level, trend, or
counts. That is unlike Agent 1, where the mock is a rough approximation
of the actual reasoning.

### 6.2 Scoring formula (deliberately simple and auditable)

```
per_incident_points = SEVERITY_WEIGHT[severity] × CONTRIBUTION_MULTIPLIER[driver_contribution.level]
raw = sum(per_incident_points for every incident)
risk_score = min(100, round(raw × (30 / time_window_days)))
```

`SEVERITY_WEIGHT`: LOW=5, MEDIUM=15, HIGH=35, CRITICAL=70 — each step up
matters a lot more than the last, so a handful of CRITICAL events don't
get diluted by a pile of LOW ones. `CONTRIBUTION_MULTIPLIER`: NONE=0.2 up
to SIGNIFICANT=1.4 — this is the same "don't automatically blame the
driver" principle from Agent 1's Rule 2, carried into the score itself:
four HIGH-severity events the driver didn't cause (defensive braking)
score meaningfully lower than four HIGH-severity events where they were
the significant contributor (`test_defensive_driving_scores_lower_than_at_fault_driving`
asserts exactly this). The `× (30 / time_window_days)` term normalizes to
a "per 30 days" basis so a driver evaluated over 7 days and one evaluated
over 90 days are comparable — without it, a longer window would look
artificially riskier just by accumulating more incidents.

Trend compares the mean weighted score of the first half of the window's
incidents (by timestamp) to the second half; below `MIN_INCIDENTS_FOR_TREND`
(4) dated incidents, it honestly reports `INSUFFICIENT_DATA` rather than
guessing. Recurring patterns group by `root_cause.cause` from Agent 1's
output and keep groups seen 2+ times, with their own (separately
computed) frequency trend.

### 6.3 The Agent 1 → Agent 2 handoff

Agent 1 didn't originally carry `timestamp` in its output (only its
input) — Agent 2 needs it to order incidents in time, so it was added as
an optional field on `IncidentOutput`, set by the agent (not the LLM)
directly from the input, the same way `incident_id` mismatches are
corrected. Everything else Agent 2 needs was already there:
`severity`, `driver_contribution`, `root_cause`, `confidence`.

Deliberately, **neither agent** decides which incidents belong to which
driver — that association is the caller's responsibility
(`scripts/analyze_driver.py` shows the full shape: raw incidents in, run
each through Agent 1 independently, group the results by driver, hand
that list to Agent 2). Agent 1 has no concept of "driver 102"; it
analyzes one incident at a time, which is what keeps it reusable outside
a fleet-management context too.

```python
analyzed = [incident_agent.analyze(raw) for raw in driver_102_raw_incidents]
driver_risk = driver_agent.analyze(DriverRiskInput(
    driver_id="102", time_window_days=30, incidents=analyzed,
))
```

`test_pipeline_integration.py` runs exactly this, starting from raw
incident dicts (not hand-built `IncidentOutput`), which is what actually
proves the two agents connect rather than each working in isolation.

### 6.4 Path to Agent 3 (Fleet Risk Analyst)

`DriverRiskOutput` is the next handoff — Agent 3 will consume a list of
these (one per driver across the fleet) the same way Agent 2 consumed a
list of `IncidentOutput`. The same split carries over even more cleanly
there: "which behaviours are trending up fleet-wide, is there a
geographic or time-of-day cluster" is almost entirely aggregation over
already-computed `risk_score`/`trend`/`incident_breakdown` fields, so
Agent 3's own stats engine is a natural next file in `stats/`, not a
rewrite of this pattern.
