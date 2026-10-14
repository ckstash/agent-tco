from __future__ import annotations

import pytest

from agent_tco import (
    evaluate_paper_chain,
    evaluate_paper_structured_exhaustive,
    evaluate_paper_structured_guided,
    paper_chain_screened_ids,
    pareto_frontier,
)


CHAIN_FRONTIER = {
    "AAAAA", "AAABA", "AABAA", "AABBA", "AABBB", "BABBA", "BABBB",
}
STRUCTURED_FRONTIER = {
    "AAAAAAAA", "ABAAAAAA", "BAAAAAAA", "BAAAABAA",
    "BABAABAA", "BBAAAAAA", "BBAAABAA", "BBBAABAA",
}


def test_paper_chain_reproduces_reported_results() -> None:
    results = evaluate_paper_chain()
    frontier = pareto_frontier(results)

    assert len(results) == 243
    assert {result.config.config_id for result in frontier} == CHAIN_FRONTIER
    best = min(results, key=lambda result: result.executive_metric)
    assert best.config.config_id == "BABBB"
    assert best.acr == pytest.approx(0.893, abs=0.0005)
    assert best.expected_tco == pytest.approx(2.76, abs=0.005)

    screened = paper_chain_screened_ids()
    screened_frontier = pareto_frontier(
        [result for result in results if result.config.config_id in screened]
    )
    recovered = CHAIN_FRONTIER & {
        result.config.config_id for result in screened_frontier
    }
    assert len(screened) == 13
    assert recovered == {"AAAAA"}


def test_paper_structured_example_reproduces_reported_results() -> None:
    exhaustive = evaluate_paper_structured_exhaustive()
    guided = evaluate_paper_structured_guided()
    exact_ids = {result.config_id for result in exhaustive.frontier}
    guided_ids = {result.config_id for result in guided.frontier}

    assert exhaustive.total_space == exhaustive.evaluated == 6561
    assert exact_ids == STRUCTURED_FRONTIER
    best = min(exhaustive.results, key=lambda result: result.executive_metric)
    assert best.config_id == "BAAAABAA"
    assert best.acr == pytest.approx(0.7248227037826349)
    assert best.expected_tco == pytest.approx(4.854172509060029)
    assert guided.evaluated == 500
    assert guided.complete is False
    assert STRUCTURED_FRONTIER <= guided_ids