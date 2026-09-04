"""
Generic, domain-agnostic pipeline engine.

This module has never heard of an "incident" or a "driver" — it only knows
how to run an ordered list of AgentSpecs (types.py) against a
PipelineContext, in "single" or "map" mode, catch each spec's declared
known_errors (nothing else — an exception not in known_errors is a bug,
not an expected pipeline failure, and must not be swallowed here), and
produce a PipelineResult with a complete execution trace.

Wiring this to Fleet Safety's Agent 1 / Agent 2 happens entirely in
fleet_pipeline.py. Adding Agent 3 (or reusing this engine for an unrelated
domain) means writing one more AgentSpec elsewhere — nothing in this file
needs to change.
"""

from __future__ import annotations

import logging
import time

from .types import AgentExecutionResult, AgentSpec, AgentStatus, PipelineContext, PipelineResult

logger = logging.getLogger(__name__)

VALID_MODES = ("single", "map")


class Orchestrator:
    """Executes an ordered list of AgentSpecs (the agent registry) against
    a PipelineContext. Construct once with the registry, call run() per
    pipeline execution — same reusable shape as the agents themselves."""

    def __init__(self, agent_specs: list[AgentSpec]):
        if not agent_specs:
            raise ValueError("Orchestrator requires at least one AgentSpec.")
        names = [spec.name for spec in agent_specs]
        if len(names) != len(set(names)):
            raise ValueError(f"AgentSpec names must be unique, got: {names}")
        for spec in agent_specs:
            if spec.mode not in VALID_MODES:
                raise ValueError(
                    f"AgentSpec {spec.name!r} has mode {spec.mode!r}; expected one of {VALID_MODES}."
                )
        self.agent_specs = agent_specs

    def run(self, initial_input: object) -> PipelineResult:
        """
        Input -> stage 1 -> validate/record -> transform state -> stage 2
        -> validate/record -> ... -> PipelineResult.

        Stages run strictly in registry order (v1 has no conditional
        routing). On the first stage that fails (single mode: the one
        call failed; map mode: any item failed), execution stops — no
        later stage runs — and a FAILED PipelineResult is returned with
        the trace built so far. Nothing about a failure is hidden: it is
        always a recorded AgentExecutionResult, never a skipped item or a
        swallowed exception.
        """
        context = PipelineContext(original_input=initial_input)
        trace: list[AgentExecutionResult] = []

        for spec in self.agent_specs:
            stage_input = spec.build_input(context)

            if spec.mode == "map":
                stage_output, stage_trace, failed = self._run_map(spec, stage_input)
            else:  # "single" — the only other value VALID_MODES permits
                stage_output, stage_trace, failed = self._run_single(spec, stage_input)

            trace.extend(stage_trace)

            if failed:
                logger.warning("Pipeline halted: stage %r failed.", spec.name)
                return PipelineResult(status=AgentStatus.FAILED, trace=trace, result=None, failed_at=spec.name)

            context.results[spec.name] = stage_output

        last_stage_name = self.agent_specs[-1].name
        return PipelineResult(
            status=AgentStatus.SUCCESS,
            trace=trace,
            result=context.results[last_stage_name],
            failed_at=None,
        )

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_one(spec: AgentSpec, item_input: object, item_ref: str | None) -> AgentExecutionResult:
        """Call spec.agent.analyze() exactly once and turn the outcome
        into an AgentExecutionResult. Only spec.known_errors is caught —
        anything else propagates immediately, uncaught, so a genuine bug
        surfaces as a genuine crash rather than a quietly FAILED stage."""
        started = time.monotonic()
        try:
            output = spec.agent.analyze(item_input)
        except spec.known_errors as e:
            ended = time.monotonic()
            return AgentExecutionResult(
                agent_name=spec.name,
                status=AgentStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                item_ref=item_ref,
                started_at=started,
                ended_at=ended,
            )
        ended = time.monotonic()
        return AgentExecutionResult(
            agent_name=spec.name,
            status=AgentStatus.SUCCESS,
            output=output,
            item_ref=item_ref,
            started_at=started,
            ended_at=ended,
        )

    @classmethod
    def _run_single(
        cls, spec: AgentSpec, stage_input: object
    ) -> tuple[object | None, list[AgentExecutionResult], bool]:
        result = cls._execute_one(spec, stage_input, item_ref=None)
        if result.status == AgentStatus.FAILED:
            return None, [result], True
        return result.output, [result], False

    @classmethod
    def _run_map(
        cls, spec: AgentSpec, stage_items: object
    ) -> tuple[list | None, list[AgentExecutionResult], bool]:
        """Every item in stage_items is attempted — a failure on one item
        does not stop the rest of this stage from running, so the trace
        always reflects what actually happened to every item (this is
        what "never silently drop a failed mapped item" means in
        practice: the item still gets its own recorded attempt). Only
        after every item has been attempted does the stage report
        overall success or failure to the pipeline; on any item failure,
        the pipeline halts before the next stage."""
        stage_trace: list[AgentExecutionResult] = []
        outputs: list = []
        any_failed = False

        for item in stage_items:
            item_ref = spec.item_ref_fn(item)
            result = cls._execute_one(spec, item, item_ref=item_ref)
            stage_trace.append(result)
            if result.status == AgentStatus.FAILED:
                any_failed = True
            else:
                outputs.append(result.output)

        if any_failed:
            return None, stage_trace, True
        return outputs, stage_trace, False
