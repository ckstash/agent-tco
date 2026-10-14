"""Budgeted configuration screening for profile-based execution graphs."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Mapping, Sequence

from agent_tco.dag import DAGSimulationResult, DAGSimulator, TaskProfile


@dataclass(frozen=True)
class DAGSearchReport:
    mode: str
    results: tuple[DAGSimulationResult, ...]
    frontier: tuple[DAGSimulationResult, ...]
    total_space: int
    evaluated: int
    elapsed_seconds: float
    complete: bool
    n_runs_per_assignment: int


def dag_pareto_frontier(
    results: Sequence[DAGSimulationResult],
) -> list[DAGSimulationResult]:
    """Return the estimated ACR/E[TCO] frontier over evaluated assignments."""
    ordered = sorted(results, key=lambda result: (-result.acr, result.expected_tco, result.config_id))
    frontier: list[DAGSimulationResult] = []
    best_cost = float("inf")
    for result in ordered:
        if result.expected_tco < best_cost:
            frontier.append(result)
            best_cost = result.expected_tco
    return sorted(frontier, key=lambda result: (result.expected_tco, result.acr, result.config_id))


def budgeted_dag_search(
    simulator: DAGSimulator,
    options: Mapping[str, Sequence[TaskProfile]],
    max_evaluations: int = 500,
    n_runs_per_assignment: int = 200,
    context_classes: Sequence[str | None] | None = None,
) -> DAGSearchReport:
    """Screen a finite model-assignment space under a hard evaluation budget.

    Homogeneous anchors and single-task alternatives from the cheapest baseline
    are evaluated first. The search then expands one-coordinate neighbors of its
    current estimated frontier. It is deterministic under the simulator seed and
    carries no global-frontier guarantee.
    """
    node_ids = tuple(simulator.graph.nodes)
    if set(options) != set(node_ids) or any(not options[node_id] for node_id in node_ids):
        raise ValueError("options must provide at least one profile for every graph node")
    if max_evaluations < 1 or n_runs_per_assignment < 1:
        raise ValueError("evaluation and run budgets must be positive")
    radices = tuple(len(options[node_id]) for node_id in node_ids)
    total_space = math_product(radices)
    budget = min(total_space, max_evaluations)
    started = perf_counter()
    results: dict[tuple[int, ...], DAGSimulationResult] = {}

    def evaluate_choice(choice: tuple[int, ...]) -> None:
        if choice in results or len(results) >= budget:
            return
        assignment = {
            node_id: options[node_id][index]
            for node_id, index in zip(node_ids, choice)
        }
        config_id = "".join(assignment[node_id].profile_id for node_id in node_ids)
        results[choice] = simulator.evaluate(
            assignment, n_runs_per_assignment, context_classes, config_id
        )

    for option_index in range(max(radices)):
        evaluate_choice(tuple(min(option_index, radix-1) for radix in radices))
    baseline = tuple(radix-1 for radix in radices)
    evaluate_choice(baseline)

    priority = sorted(
        range(len(node_ids)),
        key=lambda index: -profile_spread(options[node_ids[index]]),
    )
    for index in priority:
        for option_index in range(radices[index]-1):
            choice = list(baseline)
            choice[index] = option_index
            evaluate_choice(tuple(choice))

    expanded: set[tuple[int, ...]] = set()
    while len(results) < budget:
        inverse = {id(result): choice for choice, result in results.items()}
        seeds = [
            inverse[id(result)]
            for result in dag_pareto_frontier(tuple(results.values()))
            if inverse[id(result)] not in expanded
        ]
        if not seeds:
            remaining = next(
                (choice for choice in product(*(range(radix) for radix in radices)) if choice not in results),
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
                    if len(results) >= budget:
                        break
                if len(results) >= budget:
                    break
            if len(results) >= budget:
                break

    values = tuple(results.values())
    return DAGSearchReport(
        mode="budgeted_profile_dag",
        results=values,
        frontier=tuple(dag_pareto_frontier(values)),
        total_space=total_space,
        evaluated=len(values),
        elapsed_seconds=perf_counter()-started,
        complete=len(values) == total_space,
        n_runs_per_assignment=n_runs_per_assignment,
    )


def math_product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def profile_spread(profiles: Sequence[TaskProfile]) -> float:
    quality = [profile.q for profile in profiles]
    costs = [profile.invocation_cost for profile in profiles]
    return (max(quality)-min(quality)) + (max(costs)-min(costs))
