"""
Generic orchestration types — domain-agnostic.

Nothing in this file knows about incidents, drivers, or fleet safety. It is
the vocabulary the Orchestrator (orchestrator.py) uses to describe what it
did, regardless of which agents are plugged in. The Fleet Safety-specific
wiring (what an agent needs as input, how many there are, what "Agent 1"
even means) lives entirely in fleet_pipeline.py.

Why dataclasses here instead of Pydantic, when the rest of this project
uses Pydantic everywhere: Pydantic models are for data that crosses a
validation boundary (agent input/output — IncidentInput, DriverRiskOutput,
etc.), which is exactly what this project already uses it for. These types
hold agent instances and callables (AgentSpec.agent, AgentSpec.build_input),
which aren't meaningfully validated or serialized — they're execution
plumbing, not a schema. Pydantic would need arbitrary_types_allowed just to
hold them, which buys nothing here. Dataclasses are the right tool for
"a plain structured record," which is all these are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class AgentStatus(str, Enum):
    """Outcome of a single agent execution, or of a whole pipeline run.
    Deliberately just two values for v1 — no PARTIAL/SKIPPED/RETRYING.
    Both agents already handle their own retry/repair and fallback
    internally, so from the orchestrator's point of view a call either
    produced a valid, validated output or it didn't."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


def _no_item_ref(item: Any) -> str | None:
    """Default AgentSpec.item_ref_fn: no reference available. A map-mode
    spec that wants readable trace entries (e.g. an incident_id) supplies
    its own extractor — the engine doesn't guess at domain field names."""
    return None


@dataclass
class AgentExecutionResult:
    """One row of the execution trace: exactly one call to one agent's
    .analyze() (one item, for a map stage; the one call, for a single
    stage). Always recorded — success or failure — so nothing an agent
    was asked to do ever goes unaccounted for."""

    agent_name: str
    status: AgentStatus
    output: Any = None
    error: str | None = None
    error_type: str | None = None
    item_ref: str | None = None
    started_at: float | None = None
    ended_at: float | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000


@dataclass
class PipelineContext:
    """Explicit pipeline state, passed to every AgentSpec.build_input().
    `original_input` is whatever was given to Orchestrator.run(). `results`
    accumulates each completed stage's validated output, keyed by
    AgentSpec.name, so a later stage can read from any earlier stage — not
    just the one immediately before it."""

    original_input: Any
    results: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """What Orchestrator.run() returns. `status` reflects the whole
    pipeline: SUCCESS only if every stage succeeded. `trace` is the
    complete, ordered execution history — every stage, every mapped item,
    success or failure. `result` is the last stage's output, set only on
    overall success. `failed_at` names the stage that stopped the
    pipeline, set only on failure."""

    status: AgentStatus
    trace: list[AgentExecutionResult] = field(default_factory=list)
    result: Any = None
    failed_at: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == AgentStatus.SUCCESS


@dataclass
class AgentSpec:
    """One entry in the orchestrator's agent registry — one pipeline
    stage. Adding Agent 3 later means constructing one more AgentSpec and
    appending it to the list passed to Orchestrator(); nothing about the
    Orchestrator class itself changes.

    agent:        Any existing agent instance with an .analyze(x) method
                  (FleetSafetyIncidentAnalyst, DriverRiskAnalyst, or a
                  future Agent 3+) — used exactly as it already is,
                  unmodified.
    mode:         "single" — call agent.analyze() once on build_input()'s
                  return value.
                  "map" — build_input() returns a list; agent.analyze()
                  is called once per item, and every item is attempted
                  (a failure on one item does not skip the rest of that
                  stage — see Orchestrator._run_map).
    build_input:  Given the current PipelineContext, returns this stage's
                  input (a single object for "single", a list for "map").
                  This is the "transform/pass required state" step
                  between stages.
    known_errors: Exception types this stage is expected to potentially
                  raise as an ordinary, structured failure (e.g. this
                  project's AgentError/LLMError, or Pydantic's
                  ValidationError) — these are caught and recorded in the
                  trace. Anything NOT listed here is a bug, not an
                  expected failure, and is deliberately left to propagate
                  rather than being swallowed. Defaults to an empty tuple,
                  meaning "catch nothing" — a spec must opt in explicitly.
    item_ref_fn:  For "map" stages, given one item, returns a short human
                  -readable reference for the trace (e.g. an incident_id).
                  Domain-specific by nature, so it's supplied per spec
                  rather than guessed by the engine. Defaults to "no
                  reference available".
    """

    name: str
    agent: Any
    build_input: Callable[[PipelineContext], Any]
    mode: str = "single"
    known_errors: tuple[type[BaseException], ...] = ()
    item_ref_fn: Callable[[Any], str | None] = _no_item_ref
