"""
Fleet Safety pipeline: the concrete Agent 1 -> Agent 2 wiring on top of the
generic Orchestrator (orchestrator.py). This is the ONLY file in the
orchestration package that knows what an "incident" or a "driver" is —
orchestrator.py and types.py stay fully domain-agnostic, so a future
Agent 3+ pipeline (or an entirely different domain) can reuse them without
modification.

Nothing here changes how Agent 1 or Agent 2 work internally. Both are used
exactly as before — FleetSafetyIncidentAnalyst.analyze(),
DriverRiskAnalyst.analyze() — with their own existing retry/repair and
narrative-fallback behavior completely intact. This module only decides
*when* to call them and *how* to pass state between them.

Pipeline shape (this is not a simple 1-record chain):
    FleetPipelineInput (driver_id, time_window_days, raw_incidents)
        -> Agent 1, MAP mode: analyze() once per raw incident
        -> Agent 2, SINGLE mode: analyze() once over every successfully
           analyzed incident
        -> DriverRiskOutput
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from ..agents.driver_risk_analyst import DriverRiskAnalyst
from ..agents.incident_analyst import FleetSafetyIncidentAnalyst
from ..exceptions import AgentError
from ..llm.base import LLMError
from ..schemas import DriverRiskInput
from .orchestrator import Orchestrator
from .types import AgentSpec, PipelineContext

# Exceptions Agent 1 / Agent 2 are expected to potentially raise as an
# ordinary, structured failure (bad input data, an LLM call that never
# recovered after its own retries): this project's AgentError hierarchy
# (OutputValidationError etc.), LLMError, and Pydantic's ValidationError
# (raised directly by IncidentInput(**raw)/DriverRiskInput(**raw) on
# malformed input, before either agent's own try/except even begins).
# Anything else is a real bug and is intentionally left uncaught.
_KNOWN_AGENT_ERRORS: tuple[type[BaseException], ...] = (AgentError, LLMError, ValidationError)


class FleetPipelineInput(BaseModel):
    """What orchestrator.run() takes for the Fleet Safety pipeline:
    everything needed to go from one driver's raw incident data to a full
    driver risk assessment. `raw_incidents` are intentionally plain dicts
    (matching what Agent 1 already accepts, and what
    scripts/analyze_driver.py already reads from JSON) — validation of
    each one happens per-incident, inside Agent 1, not here."""

    driver_id: str
    time_window_days: int = Field(default=30, ge=1)
    raw_incidents: list[dict]


def _incident_item_ref(item: dict) -> str | None:
    """Trace label for a mapped Agent 1 execution — the incident_id, when
    the raw incident has one. Domain-specific by nature, which is exactly
    why this lives here and not in the generic engine."""
    if isinstance(item, dict):
        return item.get("incident_id")
    return getattr(item, "incident_id", None)


def _build_incident_analyst_input(context: PipelineContext) -> list[dict]:
    pipeline_input: FleetPipelineInput = context.original_input
    return list(pipeline_input.raw_incidents)


def _build_driver_risk_input(context: PipelineContext) -> DriverRiskInput:
    pipeline_input: FleetPipelineInput = context.original_input
    analyzed_incidents = context.results["incident_analyst"]
    return DriverRiskInput(
        driver_id=pipeline_input.driver_id,
        time_window_days=pipeline_input.time_window_days,
        incidents=analyzed_incidents,
    )


def build_fleet_safety_pipeline(
    incident_agent: FleetSafetyIncidentAnalyst,
    driver_agent: DriverRiskAnalyst,
) -> Orchestrator:
    """
    Wire the existing Agent 1 and Agent 2 instances into a two-stage
    Orchestrator: Agent 1 runs once per raw incident (map), then Agent 2
    runs once over the incidents Agent 1 successfully analyzed (single).

    Callers supply already-constructed agents, so the LLM backend is
    decided entirely by the caller, not here: pass agents built with
    MockLLMClient/DriverRiskMockLLMClient for an offline run, or
    AnthropicLLMClient for production. This function doesn't inspect or
    care which — it only calls .analyze() on whatever it's given, exactly
    as scripts/analyze_driver.py already does today.
    """
    incident_analyst_spec = AgentSpec(
        name="incident_analyst",
        agent=incident_agent,
        mode="map",
        build_input=_build_incident_analyst_input,
        known_errors=_KNOWN_AGENT_ERRORS,
        item_ref_fn=_incident_item_ref,
    )
    driver_risk_spec = AgentSpec(
        name="driver_risk_analyst",
        agent=driver_agent,
        mode="single",
        build_input=_build_driver_risk_input,
        known_errors=_KNOWN_AGENT_ERRORS,
    )
    return Orchestrator([incident_analyst_spec, driver_risk_spec])
