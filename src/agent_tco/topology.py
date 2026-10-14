"""Exact compositional evaluation for structured agentic workflow topologies.

This module deliberately supports a bounded policy language rather than arbitrary
graph execution: series, eager AND, ordered fallback OR, eager OR, and bounded
retry. Component outcomes are conditionally independent under an assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Mapping, TypeAlias

from agent_tco.model import CostParams, Scenario, StepConfig


class BlockType(Enum):
    """Execution policies supported by the structured exact evaluator."""

    SERIES = "series"
    AND_EAGER = "and_eager"
    OR_FALLBACK = "or_fallback"
    OR_EAGER = "or_eager"
    RETRY_BOUNDED = "retry_bounded"


@dataclass(frozen=True)
class TaskNode:
    """A configurable task leaf identified independently of its model assignment."""

    node_id: str

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("TaskNode node_id must not be empty")


TopologyNode: TypeAlias = "TaskNode | WorkflowBlock"


@dataclass(frozen=True)
class WorkflowBlock:
    """A nested structured workflow block with explicit execution semantics."""

    block_type: BlockType
    children: tuple[TopologyNode, ...]
    max_attempts: int = 1

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("WorkflowBlock requires at least one child")
        if self.block_type == BlockType.RETRY_BOUNDED:
            if len(self.children) != 1:
                raise ValueError("RETRY_BOUNDED requires exactly one child")
            if self.max_attempts < 1:
                raise ValueError("max_attempts must be >= 1")
        elif self.max_attempts != 1:
            raise ValueError("max_attempts is only valid for RETRY_BOUNDED")

    @classmethod
    def series(cls, *children: TopologyNode) -> "WorkflowBlock":
        return cls(BlockType.SERIES, tuple(children))

    @classmethod
    def and_eager(cls, *children: TopologyNode) -> "WorkflowBlock":
        return cls(BlockType.AND_EAGER, tuple(children))

    @classmethod
    def or_fallback(cls, *children: TopologyNode) -> "WorkflowBlock":
        return cls(BlockType.OR_FALLBACK, tuple(children))

    @classmethod
    def or_eager(cls, *children: TopologyNode) -> "WorkflowBlock":
        return cls(BlockType.OR_EAGER, tuple(children))

    @classmethod
    def retry_bounded(cls, child: TopologyNode, max_attempts: int = 2) -> "WorkflowBlock":
        return cls(BlockType.RETRY_BOUNDED, (child,), max_attempts=max_attempts)


class Completion(Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


@dataclass(frozen=True)
class ExecutionState:
    completion: Completion
    assisted: bool = False
    retried: bool = False


@dataclass(frozen=True)
class OutcomeMoment:
    """Probability and probability-weighted cost for one execution state."""

    probability: float
    weighted_cost: float


@dataclass(frozen=True)
class TopologyResult:
    """Exact ACR, expected cost, and scenario probabilities for one assignment."""

    config_id: str
    assignment: Mapping[str, StepConfig]
    acr: float
    expected_tco: float
    executive_metric: float
    scenario_probabilities: Mapping[Scenario, float]
    pareto_dominated: bool = False


Summary = dict[ExecutionState, OutcomeMoment]


def collect_task_ids(node: TopologyNode) -> tuple[str, ...]:
    """Return unique task IDs in deterministic traversal order."""
    ordered: list[str] = []

    def visit(current: TopologyNode) -> None:
        if isinstance(current, TaskNode):
            if current.node_id not in ordered:
                ordered.append(current.node_id)
            return
        for child in current.children:
            visit(child)

    visit(node)
    return tuple(ordered)


def _add(summary: Summary, state: ExecutionState, probability: float, weighted_cost: float) -> None:
    if probability == 0.0:
        return
    previous = summary.get(state, OutcomeMoment(0.0, 0.0))
    summary[state] = OutcomeMoment(
        previous.probability + probability,
        previous.weighted_cost + weighted_cost,
    )


def _combine_moments(left: OutcomeMoment, right: OutcomeMoment) -> tuple[float, float]:
    probability = left.probability * right.probability
    weighted_cost = (
        right.probability * left.weighted_cost
        + left.probability * right.weighted_cost
    )
    return probability, weighted_cost


def _task_summary(step: StepConfig, review_cost: float) -> Summary:
    u = step.incomplete_prob
    cost = step.inference_cost
    result: Summary = {}
    _add(result, ExecutionState(Completion.PASS), step.q, step.q * cost)
    _add(
        result,
        ExecutionState(Completion.PASS, assisted=True),
        step.hitl_prob,
        step.hitl_prob * (cost + review_cost),
    )
    _add(
        result,
        ExecutionState(Completion.PARTIAL, assisted=True),
        step.partial_prob,
        step.partial_prob * (cost + review_cost),
    )
    _add(result, ExecutionState(Completion.FAIL), u, u * cost)
    return result


def _series(children: list[Summary]) -> Summary:
    current: Summary = {ExecutionState(Completion.PASS): OutcomeMoment(1.0, 0.0)}
    for child in children:
        updated: Summary = {}
        for left_state, left in current.items():
            if left_state.completion != Completion.PASS:
                _add(updated, left_state, left.probability, left.weighted_cost)
                continue
            for right_state, right in child.items():
                probability, weighted_cost = _combine_moments(left, right)
                state = ExecutionState(
                    right_state.completion,
                    left_state.assisted or right_state.assisted,
                    left_state.retried or right_state.retried,
                )
                _add(updated, state, probability, weighted_cost)
        current = updated
    return current


def _eager(children: list[Summary], any_pass: bool) -> Summary:
    result: Summary = {}
    for entries in product(*(child.items() for child in children)):
        states = [entry[0] for entry in entries]
        moments = [entry[1] for entry in entries]
        probability = 1.0
        weighted_cost = 0.0
        for moment in moments:
            weighted_cost = moment.probability * weighted_cost + probability * moment.weighted_cost
            probability *= moment.probability
        completions = [state.completion for state in states]
        if any_pass:
            completion = (
                Completion.PASS if Completion.PASS in completions
                else Completion.PARTIAL if Completion.PARTIAL in completions
                else Completion.FAIL
            )
        else:
            completion = (
                Completion.FAIL if Completion.FAIL in completions
                else Completion.PARTIAL if Completion.PARTIAL in completions
                else Completion.PASS
            )
        _add(
            result,
            ExecutionState(
                completion,
                any(state.assisted for state in states),
                any(state.retried for state in states),
            ),
            probability,
            weighted_cost,
        )
    return result


def _fallback(children: list[Summary]) -> Summary:
    current: Summary = {ExecutionState(Completion.FAIL): OutcomeMoment(1.0, 0.0)}
    for child in children:
        updated: Summary = {}
        for left_state, left in current.items():
            if left_state.completion != Completion.FAIL:
                _add(updated, left_state, left.probability, left.weighted_cost)
                continue
            for right_state, right in child.items():
                probability, weighted_cost = _combine_moments(left, right)
                _add(
                    updated,
                    ExecutionState(
                        right_state.completion,
                        left_state.assisted or right_state.assisted,
                        left_state.retried or right_state.retried,
                    ),
                    probability,
                    weighted_cost,
                )
        current = updated
    return current


def _retry(child: Summary, max_attempts: int) -> Summary:
    current: Summary = {ExecutionState(Completion.FAIL): OutcomeMoment(1.0, 0.0)}
    for attempt in range(max_attempts):
        updated: Summary = {}
        for left_state, left in current.items():
            if left_state.completion != Completion.FAIL:
                _add(updated, left_state, left.probability, left.weighted_cost)
                continue
            for right_state, right in child.items():
                probability, weighted_cost = _combine_moments(left, right)
                _add(
                    updated,
                    ExecutionState(
                        right_state.completion,
                        left_state.assisted or right_state.assisted,
                        left_state.retried or right_state.retried or attempt > 0,
                    ),
                    probability,
                    weighted_cost,
                )
        current = updated
    return current


def evaluate_topology(
    topology: TopologyNode,
    assignment: Mapping[str, StepConfig],
    cost_params: CostParams,
    config_id: str | None = None,
) -> TopologyResult:
    """Evaluate one assignment exactly under the structured graph semantics."""
    required = set(collect_task_ids(topology))
    missing = required - assignment.keys()
    extra = assignment.keys() - required
    if missing or extra:
        raise ValueError(f"Assignment mismatch: missing={sorted(missing)}, extra={sorted(extra)}")

    def evaluate(node: TopologyNode) -> Summary:
        if isinstance(node, TaskNode):
            return _task_summary(assignment[node.node_id], cost_params.hitl_cost_per_event)
        children = [evaluate(child) for child in node.children]
        if node.block_type == BlockType.SERIES:
            return _series(children)
        if node.block_type == BlockType.AND_EAGER:
            return _eager(children, any_pass=False)
        if node.block_type == BlockType.OR_FALLBACK:
            return _fallback(children)
        if node.block_type == BlockType.OR_EAGER:
            return _eager(children, any_pass=True)
        if node.block_type == BlockType.RETRY_BOUNDED:
            return _retry(children[0], node.max_attempts)
        raise ValueError(f"Unsupported block type: {node.block_type}")

    summary = evaluate(topology)
    scenario_probabilities = {scenario: 0.0 for scenario in Scenario}
    expected_tco = 0.0
    for state, moment in summary.items():
        expected_tco += moment.weighted_cost
        if state.completion == Completion.FAIL:
            scenario = Scenario.ABORTED
            expected_tco += moment.probability * cost_params.manual_process_cost
        elif state.completion == Completion.PARTIAL:
            scenario = Scenario.HITL_PARTIAL
        elif state.assisted:
            scenario = Scenario.COMPOUND_RECOVERY if state.retried else Scenario.HITL_SUCCESS
        else:
            scenario = Scenario.RETRY_RESOLVED if state.retried else Scenario.HAPPY_PATH
        scenario_probabilities[scenario] += moment.probability

    acr = (
        scenario_probabilities[Scenario.HAPPY_PATH]
        + scenario_probabilities[Scenario.RETRY_RESOLVED]
    )
    identifier = config_id or "".join(assignment[node_id].model_tier for node_id in collect_task_ids(topology))
    return TopologyResult(
        config_id=identifier,
        assignment=dict(assignment),
        acr=acr,
        expected_tco=expected_tco,
        executive_metric=expected_tco / acr if acr else float("inf"),
        scenario_probabilities=scenario_probabilities,
    )


def topology_acr(topology: TopologyNode, assignment: Mapping[str, StepConfig]) -> float:
    """Return autonomous reliability without review or fallback prices."""
    return evaluate_topology(topology, assignment, CostParams(0.0, 0.0, 0.0)).acr


def birnbaum_importance(
    topology: TopologyNode,
    assignment: Mapping[str, StepConfig],
    node_id: str,
) -> float:
    """Return the finite binary-component Birnbaum importance of one task.

    The selected task is replaced by an autonomous component with reliability
    one and then zero. This is a structural screening score; finite model moves
    must still be evaluated with their costs.
    """
    if node_id not in assignment:
        raise ValueError(f"Unknown node_id: {node_id}")
    base = assignment[node_id]

    def replacement(q: float) -> StepConfig:
        return StepConfig(base.step_index, base.model_tier, q, 0.0, base.inference_cost, 0.0)

    high = dict(assignment)
    low = dict(assignment)
    high[node_id] = replacement(1.0)
    low[node_id] = replacement(0.0)
    return topology_acr(topology, high) - topology_acr(topology, low)