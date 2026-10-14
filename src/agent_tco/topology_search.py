"""Assignment search for structured workflow topologies."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Mapping, Sequence

from agent_tco.model import CostParams, StepConfig
from agent_tco.topology import (
    TopologyNode,
    TopologyResult,
    birnbaum_importance,
    collect_task_ids,
    evaluate_topology,
)


@dataclass(frozen=True)
class SearchReport:
    """Results and audit metadata for one assignment search."""

    mode: str
    results: tuple[TopologyResult, ...]
    frontier: tuple[TopologyResult, ...]
    total_space: int
    evaluated: int
    elapsed_seconds: float
    complete: bool


def topology_pareto_frontier(results: Sequence[TopologyResult]) -> list[TopologyResult]:
    """Return the two-objective frontier in O(n log n) time."""
    ordered = sorted(results, key=lambda result: (-result.acr, result.expected_tco, result.config_id))
    frontier: list[TopologyResult] = []
    best_cost = float("inf")
    for result in ordered:
        if result.expected_tco < best_cost:
            frontier.append(result)
            best_cost = result.expected_tco
    return sorted(frontier, key=lambda result: (result.expected_tco, result.acr, result.config_id))


def _validate_options(
    topology: TopologyNode,
    options: Mapping[str, Sequence[StepConfig]],
) -> tuple[str, ...]:
    node_ids = collect_task_ids(topology)
    if set(node_ids) != set(options):
        raise ValueError(
            f"Option mismatch: required={sorted(node_ids)}, supplied={sorted(options)}"
        )
    if any(not options[node_id] for node_id in node_ids):
        raise ValueError("Every task requires at least one model option")
    return node_ids


def exhaustive_topology_search(
    topology: TopologyNode,
    options: Mapping[str, Sequence[StepConfig]],
    cost_params: CostParams,
) -> SearchReport:
    """Evaluate the complete Cartesian assignment space exactly."""
    node_ids = _validate_options(topology, options)
    started = perf_counter()
    results = []
    for choices in product(*(options[node_id] for node_id in node_ids)):
        assignment = dict(zip(node_ids, choices))
        config_id = "".join(step.model_tier for step in choices)
        results.append(evaluate_topology(topology, assignment, cost_params, config_id))
    elapsed = perf_counter() - started
    return SearchReport(
        mode="exhaustive",
        results=tuple(results),
        frontier=tuple(topology_pareto_frontier(results)),
        total_space=len(results),
        evaluated=len(results),
        elapsed_seconds=elapsed,
        complete=True,
    )


def guided_topology_search(
    topology: TopologyNode,
    options: Mapping[str, Sequence[StepConfig]],
    cost_params: CostParams,
    max_evaluations: int = 500,
) -> SearchReport:
    """Return a deterministic budgeted frontier approximation.

    The search starts from homogeneous anchors, ranks tasks by topology-aware
    Birnbaum importance, and repeatedly evaluates one-coordinate neighbors of
    the current frontier. It is intentionally heuristic and carries no global
    frontier guarantee.
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be >= 1")
    node_ids = _validate_options(topology, options)
    radices = tuple(len(options[node_id]) for node_id in node_ids)
    total_space = 1
    for radix in radices:
        total_space *= radix
    budget = min(max_evaluations, total_space)
    started = perf_counter()
    results_by_choice: dict[tuple[int, ...], TopologyResult] = {}

    def evaluate_choice(choice: tuple[int, ...]) -> None:
        if choice in results_by_choice or len(results_by_choice) >= budget:
            return
        steps = tuple(options[node_id][index] for node_id, index in zip(node_ids, choice))
        assignment = dict(zip(node_ids, steps))
        config_id = "".join(step.model_tier for step in steps)
        results_by_choice[choice] = evaluate_topology(topology, assignment, cost_params, config_id)

    anchors = []
    for option_index in range(max(radices)):
        anchors.append(tuple(min(option_index, radix - 1) for radix in radices))
    anchors.extend((tuple(0 for _ in node_ids), tuple(radix - 1 for radix in radices)))
    for anchor in anchors:
        evaluate_choice(anchor)

    baseline_choice = tuple(radix - 1 for radix in radices)
    baseline = {node_id: options[node_id][index] for node_id, index in zip(node_ids, baseline_choice)}
    priority = sorted(
        range(len(node_ids)),
        key=lambda index: (-birnbaum_importance(topology, baseline, node_ids[index]), index),
    )
    for index in priority:
        for option_index in range(radices[index] - 1):
            choice = list(baseline_choice)
            choice[index] = option_index
            evaluate_choice(tuple(choice))

    expanded: set[tuple[int, ...]] = set()
    while len(results_by_choice) < budget:
        inverse = {id(result): choice for choice, result in results_by_choice.items()}
        frontier = topology_pareto_frontier(tuple(results_by_choice.values()))
        seeds = [inverse[id(result)] for result in frontier if inverse[id(result)] not in expanded]
        if not seeds:
            remaining = next(
                (choice for choice in product(*(range(radix) for radix in radices))
                 if choice not in results_by_choice),
                None,
            )
            if remaining is None:
                break
            evaluate_choice(remaining)
            continue
        for seed in seeds:
            expanded.add(seed)
            for index in priority:
                for option_index in range(radices[index]):
                    if option_index == seed[index]:
                        continue
                    neighbor = list(seed)
                    neighbor[index] = option_index
                    evaluate_choice(tuple(neighbor))
                    if len(results_by_choice) >= budget:
                        break
                if len(results_by_choice) >= budget:
                    break
            if len(results_by_choice) >= budget:
                break

    results = tuple(results_by_choice.values())
    return SearchReport(
        mode="guided",
        results=results,
        frontier=tuple(topology_pareto_frontier(results)),
        total_space=total_space,
        evaluated=len(results),
        elapsed_seconds=perf_counter() - started,
        complete=len(results) == total_space,
    )