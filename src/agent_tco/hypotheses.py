"""Reproducible stylized computational tests for the five paper hypotheses.

These experiments test implications of the declared workflow model. They are not
field validation: profile parameters are constructed, and deployment claims require
measured trajectories from the target agent and workload.
"""
from __future__ import annotations

from typing import Any

from agent_tco.dag import DAGNode, DAGSimulator, FailureDomain, SuccessPolicy, TaskProfile, WorkflowGraph
from agent_tco.model import CostParams, StepConfig
from agent_tco.topology import TaskNode, WorkflowBlock, birnbaum_importance, evaluate_topology
from agent_tco.topology_search import topology_pareto_frontier


def _step(index: int, tier: str, q: float, cost: float, h: float = 0.0) -> StepConfig:
    return StepConfig(index, tier, q, h, cost)


def _point(label: str, result: Any, **parameters: Any) -> dict[str, Any]:
    return {
        "label": label,
        "acr": result.acr,
        "expected_tco": result.expected_tco,
        "parameters": parameters,
    }


def _frontier_slopes(points: list[dict[str, Any]]) -> list[float]:
    ordered = sorted(points, key=lambda row: row["acr"])
    return [
        (right["expected_tco"] - left["expected_tco"]) / (right["acr"] - left["acr"])
        for left, right in zip(ordered, ordered[1:])
    ]


def _diminishing_returns() -> dict[str, Any]:
    topology = WorkflowBlock.series(*(TaskNode(f"s{i}") for i in range(1, 4)))
    costs = CostParams(0.0, 0.0, 0.0)

    def family(prices: tuple[float, ...]) -> tuple[list[dict[str, Any]], list[float]]:
        reliabilities = (0.70, 0.82, 0.90, 0.95)
        rows = []
        results = []
        for tier, (q, price) in enumerate(zip(reliabilities, prices), start=1):
            assignment = {
                f"s{i}": _step(i, str(tier), q, price) for i in range(1, 4)
            }
            result = evaluate_topology(topology, assignment, costs, f"tier-{tier}")
            results.append(result)
            rows.append(_point(f"tier-{tier}", result, q=q, per_step_price=price))
        frontier_ids = {row.config_id for row in topology_pareto_frontier(results)}
        frontier = [row for row in rows if row["label"] in frontier_ids]
        return frontier, _frontier_slopes(frontier)

    convex, convex_slopes = family((0.05, 0.15, 0.60, 2.40))
    nonconvex, nonconvex_slopes = family((0.05, 0.14, 0.18, 0.19))
    return {
        "hypothesis": "H1_diminishing_returns",
        "evaluation_mode": "exact",
        "condition": "fixed chain topology and sufficiently convex per-step capability prices",
        "observations": {"convex_frontier": convex, "convex_slopes": convex_slopes,
                         "nonconvex_price_control": nonconvex,
                         "nonconvex_price_control_slopes": nonconvex_slopes},
        "supported_in_stylized_case": all(b > a for a, b in zip(convex_slopes, convex_slopes[1:])),
        "boundary_case": "A non-convex price control reverses the slopes, so diminishing returns are not unconditional.",
    }


def _finite_ceiling() -> dict[str, Any]:
    q = 0.90
    costs = CostParams(0.0, 0.0, 0.0)
    chain_rows = []
    for length in range(1, 7):
        topology = WorkflowBlock.series(*(TaskNode(f"s{i}") for i in range(1, length + 1)))
        assignment = {f"s{i}": _step(i, "best", q, 0.0) for i in range(1, length + 1)}
        result = evaluate_topology(topology, assignment, costs, f"K={length}")
        chain_rows.append(_point(f"K={length}", result, length=length, analytical=q ** length))
    redundant = evaluate_topology(
        WorkflowBlock.or_fallback(TaskNode("primary"), TaskNode("fallback")),
        {"primary": _step(1, "best", q, 0.0), "fallback": _step(2, "best", q, 0.0)},
        costs,
        "independent-fallback",
    )
    return {
        "hypothesis": "H2_finite_reliability_ceiling",
        "evaluation_mode": "exact",
        "condition": "all required best-tier passage probabilities are below one",
        "observations": {"chain_ceilings": chain_rows,
                         "independent_or_ceiling": _point("independent-fallback", redundant)},
        "supported_in_stylized_case": (
            all(row["acr"] < 1.0 for row in chain_rows)
            and all(b["acr"] < a["acr"] for a, b in zip(chain_rows, chain_rows[1:]))
            and redundant.acr > chain_rows[0]["acr"]
        ),
        "boundary_case": "A perfect required component (q=1) would not lower the chain ceiling.",
    }


def _redundancy_dependence(n_runs: int, seed: int) -> dict[str, Any]:
    costs = CostParams(0.0, 0.0, 0.0)
    primary = evaluate_topology(TaskNode("primary"), {"primary": _step(1, "P", 0.80, 1.0)}, costs)
    independent = evaluate_topology(
        WorkflowBlock.or_fallback(TaskNode("primary"), TaskNode("fallback")),
        {"primary": _step(1, "P", 0.80, 1.0), "fallback": _step(2, "F", 0.60, 2.0)},
        costs,
    )
    # A 0.10 run-level shared outage and conditional q values preserve marginal
    # branch reliabilities of 0.80 and 0.60 while inducing positive dependence.
    graph = WorkflowGraph(
        "dependent-redundancy",
        {node_id: DAGNode(node_id) for node_id in ("primary", "fallback")},
        (), ("primary", "fallback"), ("primary", "fallback"),
        success_policy=SuccessPolicy.ANY,
        failure_domains=(FailureDomain("shared", 0.10, frozenset({"primary", "fallback"})),),
    )
    profiles = {
        "primary": TaskProfile("P", 0.80 / 0.90, 0.0, 0.0, 1.0, failure_domains=("shared",)),
        "fallback": TaskProfile("F", 0.60 / 0.90, 0.0, 0.0, 2.0, failure_domains=("shared",)),
    }
    dependent = DAGSimulator(graph, costs, seed=seed).evaluate(profiles, n_runs=n_runs)
    independent_gain = independent.acr - primary.acr
    dependent_gain = dependent.acr - primary.acr
    return {
        "hypothesis": "H3_redundancy_under_dependent_errors",
        "evaluation_mode": "exact_plus_seeded_monte_carlo",
        "condition": "fallback covers some primary-failure states; shared errors are explicit",
        "observations": {
            "primary": _point("primary", primary),
            "independent_ordered_fallback": _point("independent", independent),
            "dependent_eager_or": _point("dependent", dependent, n_runs=n_runs, seed=seed),
            "independent_gain": independent_gain,
            "dependent_gain": dependent_gain,
            "independent_expected_fallback_cost": (1.0 - 0.80) * 2.0,
        },
        "supported_in_stylized_case": 0.0 < dependent_gain < independent_gain,
        "boundary_case": "Perfectly shared failure states can eliminate the reliability benefit.",
    }


def _labor_reordering() -> dict[str, Any]:
    topology = TaskNode("task")
    options = {
        "review-heavy": _step(1, "review-heavy", 0.75, 0.05, 0.20),
        "review-light": _step(1, "review-light", 0.85, 0.80, 0.05),
    }
    rows = []
    for wage in (0.0, 10.0, 20.0, 40.0):
        results = [
            evaluate_topology(topology, {"task": option}, CostParams(wage, 0.25, 0.0), label)
            for label, option in options.items()
        ]
        frontier = {result.config_id for result in topology_pareto_frontier(results)}
        for result in results:
            rows.append(_point(result.config_id, result, labor_rate=wage,
                               on_frontier=result.config_id in frontier))
    by_label = {label: [row for row in rows if row["label"] == label] for label in options}
    acr_invariant = all(len({row["acr"] for row in values}) == 1 for values in by_label.values())
    return {
        "hypothesis": "H4_labor_rate_reordering",
        "evaluation_mode": "exact",
        "condition": "wages change while model behavior and review duration remain fixed",
        "observations": {"points": rows, "analytical_break_even_labor_rate": 20.0},
        "supported_in_stylized_case": (
            acr_invariant
            and by_label["review-heavy"][0]["parameters"]["on_frontier"]
            and not by_label["review-heavy"][-1]["parameters"]["on_frontier"]
        ),
        "boundary_case": "If wages alter reviewer or model behavior, trajectories must be re-estimated.",
    }


def _importance_review_alignment() -> dict[str, Any]:
    topology = WorkflowBlock.series(
        WorkflowBlock.or_eager(TaskNode("a"), TaskNode("b")), TaskNode("c")
    )
    costs = CostParams(80.0, 0.25, 0.0)
    baseline = {
        "a": _step(1, "base", 0.80, 0.05, 0.02),
        "b": _step(2, "base", 0.50, 0.05, 0.05),
        "c": _step(3, "base", 0.90, 0.05, 0.08),
    }
    importance = {node_id: birnbaum_importance(topology, baseline, node_id) for node_id in baseline}

    def upgraded(node_id: str, extra_cost: float, review_reduction: float) -> Any:
        assignment = dict(baseline)
        old = assignment[node_id]
        assignment[node_id] = _step(old.step_index, "upgrade", old.q + 0.05,
                                    old.inference_cost + extra_cost,
                                    old.hitl_prob - review_reduction)
        return evaluate_topology(topology, assignment, costs, f"upgrade-{node_id}")

    base = evaluate_topology(topology, baseline, costs, "baseline")
    low = upgraded("b", 0.10, 0.03)
    aligned = upgraded("c", 0.10, 0.08)
    costly = upgraded("c", 5.00, 0.08)
    return {
        "hypothesis": "H5_importance_review_alignment",
        "evaluation_mode": "exact",
        "condition": "compare finite upgrades using importance, reliability increment, review reduction, and price",
        "observations": {
            "birnbaum_importance": importance,
            "baseline": _point("baseline", base),
            "low_importance_upgrade": _point("upgrade-b", low),
            "aligned_upgrade": _point("upgrade-c", aligned),
            "costly_aligned_control": _point("costly-upgrade-c", costly),
        },
        "supported_in_stylized_case": (
            aligned.acr > low.acr and aligned.expected_tco < low.expected_tco
            and aligned.acr > base.acr and aligned.expected_tco < base.expected_tco
        ),
        "boundary_case": "The costly aligned control improves ACR but not TCO, so alignment alone is insufficient.",
    }


def run_stylized_hypothesis_tests(n_runs: int = 50_000, seed: int = 20260924) -> dict[str, Any]:
    """Run the five constructed hypothesis checks and return JSON-ready results.

    Args:
        n_runs: Monte Carlo runs for the dependent-error comparison in H3.
        seed: Deterministic base seed for H3.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    experiments = [
        _diminishing_returns(),
        _finite_ceiling(),
        _redundancy_dependence(n_runs, seed),
        _labor_reordering(),
        _importance_review_alignment(),
    ]
    return {
        "schema_version": "1.0",
        "evidence_scope": "stylized_computational_test_not_field_validation",
        "seed": seed,
        "experiments": experiments,
        "all_supported_in_stylized_cases": all(row["supported_in_stylized_case"] for row in experiments),
    }