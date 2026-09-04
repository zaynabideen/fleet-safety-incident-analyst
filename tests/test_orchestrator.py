"""
Tests for the generic Orchestrator engine (orchestration/orchestrator.py),
deliberately independent of the Fleet Safety domain. These use trivial fake
agents so a failure here can only mean a bug in the engine itself, never in
Agent 1/Agent 2's own reasoning or schemas. The domain-specific pipeline
(Agent 1 -> Agent 2, wired through this same engine) is covered separately
in test_fleet_pipeline.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from fleet_safety.orchestration.orchestrator import Orchestrator
from fleet_safety.orchestration.types import AgentSpec, AgentStatus


class FakeAgentError(Exception):
    """Stand-in for a domain AgentError, used only in these engine tests."""


class UppercaseAgent:
    """Trivial single-mode agent."""

    def analyze(self, text):
        return text.upper()


class DoubleAgent:
    """Trivial map-mode agent. Raises FakeAgentError on a negative input,
    to exercise the failure path without any domain dependency."""

    def analyze(self, n):
        if n < 0:
            raise FakeAgentError(f"negative input not allowed: {n}")
        return n * 2


class BrokenAgent:
    """Raises a plain, undeclared exception — must NOT be caught by the
    orchestrator, since it is not in known_errors. Simulates a real bug
    rather than an expected domain/LLM/validation failure."""

    def analyze(self, x):
        raise RuntimeError("unexpected programming error")


def _single_spec(name="uppercase"):
    return AgentSpec(
        name=name,
        agent=UppercaseAgent(),
        mode="single",
        build_input=lambda ctx: ctx.original_input,
        known_errors=(FakeAgentError,),
    )


def _double_spec(name="double"):
    return AgentSpec(
        name=name,
        agent=DoubleAgent(),
        mode="map",
        build_input=lambda ctx: ctx.original_input,
        known_errors=(FakeAgentError,),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_orchestrator_rejects_empty_spec_list():
    with pytest.raises(ValueError):
        Orchestrator([])


def test_orchestrator_rejects_duplicate_spec_names():
    with pytest.raises(ValueError):
        Orchestrator([_single_spec("dup"), _single_spec("dup")])


def test_orchestrator_rejects_unknown_mode():
    bad_spec = AgentSpec(
        name="bad", agent=UppercaseAgent(), mode="parallel",
        build_input=lambda ctx: ctx.original_input,
    )
    with pytest.raises(ValueError):
        Orchestrator([bad_spec])


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

def test_successful_single_stage_execution():
    orchestrator = Orchestrator([_single_spec()])
    result = orchestrator.run("hello")

    assert result.status == AgentStatus.SUCCESS
    assert result.result == "HELLO"
    assert len(result.trace) == 1
    assert result.trace[0].agent_name == "uppercase"
    assert result.trace[0].status == AgentStatus.SUCCESS
    assert result.trace[0].output == "HELLO"
    assert result.trace[0].duration_ms is not None


def test_successful_map_execution_single_item():
    orchestrator = Orchestrator([_double_spec()])
    result = orchestrator.run([5])

    assert result.status == AgentStatus.SUCCESS
    assert result.result == [10]
    assert len(result.trace) == 1


def test_successful_map_execution_multiple_items():
    orchestrator = Orchestrator([_double_spec()])
    result = orchestrator.run([1, 2, 3, 4])

    assert result.status == AgentStatus.SUCCESS
    assert result.result == [2, 4, 6, 8]
    assert len(result.trace) == 4
    assert [r.status for r in result.trace] == [AgentStatus.SUCCESS] * 4


def test_execution_ordering_across_multiple_stages():
    order = []

    class TrackingAgent:
        def __init__(self, tag):
            self.tag = tag

        def analyze(self, x):
            order.append(self.tag)
            return x

    spec_a = AgentSpec(name="a", agent=TrackingAgent("a"), mode="single",
                        build_input=lambda ctx: ctx.original_input)
    spec_b = AgentSpec(name="b", agent=TrackingAgent("b"), mode="single",
                        build_input=lambda ctx: ctx.results["a"])
    spec_c = AgentSpec(name="c", agent=TrackingAgent("c"), mode="single",
                        build_input=lambda ctx: ctx.results["b"])

    orchestrator = Orchestrator([spec_a, spec_b, spec_c])
    result = orchestrator.run("x")

    assert order == ["a", "b", "c"]
    assert [r.agent_name for r in result.trace] == ["a", "b", "c"]
    assert result.status == AgentStatus.SUCCESS
    assert result.result == "x"


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def test_map_execution_processes_every_item_even_with_a_failure_in_the_middle():
    orchestrator = Orchestrator([_double_spec()])
    result = orchestrator.run([1, -2, 3])

    # every item must appear in the trace — a failure must not cause the
    # remaining items to be silently skipped.
    assert len(result.trace) == 3
    assert result.trace[0].status == AgentStatus.SUCCESS
    assert result.trace[1].status == AgentStatus.FAILED
    assert result.trace[2].status == AgentStatus.SUCCESS
    assert result.status == AgentStatus.FAILED


def test_single_stage_failure_marks_pipeline_failed():
    class AlwaysFailsAgent:
        def analyze(self, x):
            raise FakeAgentError("boom")

    spec = AgentSpec(name="fails", agent=AlwaysFailsAgent(), mode="single",
                      build_input=lambda ctx: ctx.original_input,
                      known_errors=(FakeAgentError,))
    orchestrator = Orchestrator([spec])
    result = orchestrator.run("x")

    assert result.status == AgentStatus.FAILED
    assert result.failed_at == "fails"
    assert result.trace[0].status == AgentStatus.FAILED
    assert result.result is None


def test_halt_on_failure_stops_downstream_stages():
    calls = []

    class RecordingAgent:
        def analyze(self, x):
            calls.append(x)
            return x

    downstream_spec = AgentSpec(
        name="downstream", agent=RecordingAgent(), mode="single",
        build_input=lambda ctx: ctx.results.get("double"),
        known_errors=(FakeAgentError,),
    )

    orchestrator = Orchestrator([_double_spec(), downstream_spec])
    result = orchestrator.run([-1])

    assert result.status == AgentStatus.FAILED
    assert result.failed_at == "double"
    assert calls == []  # downstream never ran
    assert [r.agent_name for r in result.trace] == ["double"]


def test_failure_appears_in_trace_with_error_details():
    orchestrator = Orchestrator([_double_spec()])
    result = orchestrator.run([-5])

    failed_entries = [r for r in result.trace if r.status == AgentStatus.FAILED]
    assert len(failed_entries) == 1
    assert failed_entries[0].error_type == "FakeAgentError"
    assert "negative input" in failed_entries[0].error


def test_pipeline_result_has_no_final_result_on_failure():
    orchestrator = Orchestrator([_double_spec()])
    result = orchestrator.run([-1])

    assert result.result is None
    assert result.status == AgentStatus.FAILED


def test_unexpected_error_is_not_swallowed():
    """An exception NOT declared in known_errors is a bug, not an expected
    pipeline failure — it must propagate, not be caught and hidden inside
    a FAILED AgentExecutionResult."""
    spec = AgentSpec(name="broken", agent=BrokenAgent(), mode="single",
                      build_input=lambda ctx: ctx.original_input,
                      known_errors=(FakeAgentError,))
    orchestrator = Orchestrator([spec])

    with pytest.raises(RuntimeError, match="unexpected programming error"):
        orchestrator.run("x")


# ---------------------------------------------------------------------------
# item_ref
# ---------------------------------------------------------------------------

def test_item_ref_for_mapped_executions():
    class PassthroughAgent:
        def analyze(self, item):
            return item["value"]

    spec = AgentSpec(
        name="passthrough", agent=PassthroughAgent(), mode="map",
        build_input=lambda ctx: ctx.original_input,
        item_ref_fn=lambda item: item.get("ref"),
    )
    orchestrator = Orchestrator([spec])
    items = [{"ref": "A-1", "value": 1}, {"ref": "A-2", "value": 2}]
    result = orchestrator.run(items)

    assert [r.item_ref for r in result.trace] == ["A-1", "A-2"]
    assert result.status == AgentStatus.SUCCESS
    assert result.result == [1, 2]


def test_item_ref_defaults_to_none_when_not_provided():
    orchestrator = Orchestrator([_double_spec()])
    result = orchestrator.run([1, 2])

    assert all(r.item_ref is None for r in result.trace)


def test_single_mode_item_ref_is_none():
    orchestrator = Orchestrator([_single_spec()])
    result = orchestrator.run("hello")

    assert result.trace[0].item_ref is None
