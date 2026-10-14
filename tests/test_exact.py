from __future__ import annotations

import pytest

from agent_tco import CostParams, GridSearch, StepConfig, WorkflowConfig, evaluate_exact


def test_exact_acr_matches_closed_form() -> None:
    config = WorkflowConfig(
        "two_step",
        [
            StepConfig(1, "X", q=0.70, hitl_prob=0.10, partial_prob=0.05, inference_cost=0.02),
            StepConfig(2, "X", q=0.80, hitl_prob=0.05, partial_prob=0.05, inference_cost=0.03),
        ],
    )
    result = evaluate_exact(config, CostParams(60, 0.25, 4.50))
    expected = 0.70 * (1 + 0.15) * 0.80 * (1 + 0.10)
    assert result.acr == pytest.approx(expected)
    assert sum(s.probability for s in result.scenario_stats.values()) == pytest.approx(1.0)


def test_monte_carlo_acr_tracks_exact_value() -> None:
    config = WorkflowConfig(
        "two_step_mc",
        [
            StepConfig(1, "X", q=0.70, hitl_prob=0.10, partial_prob=0.05, inference_cost=0.02),
            StepConfig(2, "X", q=0.80, hitl_prob=0.05, partial_prob=0.05, inference_cost=0.03),
        ],
    )
    costs = CostParams(60, 0.25, 4.50)
    exact = evaluate_exact(config, costs)
    simulated = GridSearch(costs, n_runs=20_000, seed=7).evaluate(config)
    assert simulated.acr == pytest.approx(exact.acr, abs=0.015)
    assert simulated.acr_ci_lower <= exact.acr <= simulated.acr_ci_upper


def test_exact_hitl_success_continues_to_downstream_step() -> None:
    config = WorkflowConfig(
        "assisted_then_autonomous",
        [
            StepConfig(1, "X", q=0.0, hitl_prob=1.0, inference_cost=0.02),
            StepConfig(2, "X", q=1.0, hitl_prob=0.0, inference_cost=0.03),
        ],
    )
    costs = CostParams(60, 0.25, 4.50)
    result = evaluate_exact(config, costs)

    assert result.scenario_stats[next(s for s in result.scenario_stats if s.value == "i3")].probability == 1.0
    assert result.expected_tco == pytest.approx(0.02 + 15.0 + 0.03)