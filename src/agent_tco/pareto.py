from __future__ import annotations

from agent_tco.grid import GridResult


def dominates(a: GridResult, b: GridResult) -> bool:
    """Return True if a Pareto-dominates b.

    a dominates b when a.acr >= b.acr and a.expected_tco <= b.expected_tco,
    with at least one strict inequality.
    """
    better_acr = a.acr >= b.acr
    better_tco = a.expected_tco <= b.expected_tco
    strictly_better = (a.acr > b.acr) or (a.expected_tco < b.expected_tco)
    return better_acr and better_tco and strictly_better


def pareto_frontier(results: list[GridResult]) -> list[GridResult]:
    """Return the non-dominated subset of results.

    Sets pareto_dominated=True on every dominated GridResult in-place.
    """
    for r in results:
        r.pareto_dominated = False

    frontier = []
    for candidate in results:
        dominated = any(dominates(other, candidate) for other in results if other is not candidate)
        if dominated:
            candidate.pareto_dominated = True
        else:
            frontier.append(candidate)
    return frontier


def executive_metric(expected_tco: float, acr: float) -> float:
    """Return the descriptive cost-effectiveness ratio E[TCO]/ACR."""
    if acr == 0:
        return float("inf")
    return expected_tco / acr


def rank_by_executive_metric(results: list[GridResult]) -> list[GridResult]:
    """Return results sorted by the descriptive cost-effectiveness ratio."""
    return sorted(results, key=lambda r: r.executive_metric)
