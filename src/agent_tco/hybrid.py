"""Facade selecting exact structured evaluation or approximate graph simulation."""
from __future__ import annotations

from typing import Mapping, Sequence, overload

from agent_tco.dag import DAGSimulationResult, DAGSimulator, TaskProfile, WorkflowGraph
from agent_tco.model import CostParams, StepConfig
from agent_tco.topology import TopologyNode, TopologyResult, evaluate_topology


@overload
def evaluate_hybrid(
    workflow: TopologyNode,
    assignment: Mapping[str, StepConfig],
    cost_params: CostParams,
    *,
    config_id: str | None = None,
    n_runs: int = 1_000,
    seed: int = 42,
    context_classes: Sequence[str | None] | None = None,
) -> TopologyResult: ...


@overload
def evaluate_hybrid(
    workflow: WorkflowGraph,
    assignment: Mapping[str, TaskProfile],
    cost_params: CostParams,
    *,
    config_id: str | None = None,
    n_runs: int = 1_000,
    seed: int = 42,
    context_classes: Sequence[str | None] | None = None,
) -> DAGSimulationResult: ...


def evaluate_hybrid(
    workflow,
    assignment,
    cost_params: CostParams,
    *,
    config_id: str | None = None,
    n_runs: int = 1_000,
    seed: int = 42,
    context_classes: Sequence[str | None] | None = None,
):
    """Use exact block composition when possible and DAG Monte Carlo otherwise."""
    if isinstance(workflow, WorkflowGraph):
        return DAGSimulator(workflow, cost_params, seed).evaluate(
            assignment, n_runs, context_classes, config_id
        )
    return evaluate_topology(workflow, assignment, cost_params, config_id)
