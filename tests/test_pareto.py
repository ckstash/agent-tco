from __future__ import annotations

import pytest

from agent_tco.grid import GridResult, GridSearch
from agent_tco.model import CostParams, WorkflowConfig
from agent_tco.pareto import dominates, executive_metric, pareto_frontier, rank_by_executive_metric


def _make_result(acr: float, tco: float, config_id: str = "x") -> GridResult:
    """Create a minimal GridResult with given ACR and E[TCO] for Pareto tests."""
    from agent_tco.model import StepConfig
    config = WorkflowConfig(config_id, [
        StepConfig(step_index=1, model_tier="C", acr=acr, hitl_prob=0.0, inference_cost=0.001)
    ])
    return GridResult(
        config=config,
        n_runs=100,
        scenario_stats={},
        acr=acr,
        expected_tco=tco,
        executive_metric=tco / acr if acr > 0 else float("inf"),
    )


def test_dominates_both_better() -> None:
    """a with higher ACR and lower TCO dominates b."""
    a = _make_result(acr=0.9, tco=1.0, config_id="a")
    b = _make_result(acr=0.7, tco=2.0, config_id="b")
    assert dominates(a, b) is True
    assert dominates(b, a) is False


def test_dominates_one_equal_one_better() -> None:
    """a with equal ACR and strictly lower TCO dominates b."""
    a = _make_result(acr=0.8, tco=1.0, config_id="a")
    b = _make_result(acr=0.8, tco=1.5, config_id="b")
    assert dominates(a, b) is True


def test_does_not_dominate_tradeoff() -> None:
    """a with higher ACR but higher TCO does not dominate b."""
    a = _make_result(acr=0.9, tco=2.0, config_id="a")
    b = _make_result(acr=0.7, tco=1.0, config_id="b")
    assert dominates(a, b) is False
    assert dominates(b, a) is False


def test_pareto_frontier_excludes_dominated(
    financial_agent_configs: list[WorkflowConfig],
    cost_params: CostParams,
) -> None:
    """j3 is not in the Pareto frontier (it is dominated by j9)."""
    gs = GridSearch(cost_params=cost_params, n_runs=300, seed=42)
    results = gs.evaluate_all(financial_agent_configs)
    frontier = pareto_frontier(results)
    frontier_ids = {r.config.config_id for r in frontier}
    assert "j3" not in frontier_ids


def test_pareto_frontier_sets_dominated_flag(
    financial_agent_configs: list[WorkflowConfig],
    cost_params: CostParams,
) -> None:
    """pareto_frontier() sets pareto_dominated=True on dominated configs in-place."""
    gs = GridSearch(cost_params=cost_params, n_runs=300, seed=42)
    results = gs.evaluate_all(financial_agent_configs)
    pareto_frontier(results)
    j3 = next(r for r in results if r.config.config_id == "j3")
    assert j3.pareto_dominated is True


def test_rank_by_executive_metric_order(
    financial_agent_configs: list[WorkflowConfig],
    cost_params: CostParams,
) -> None:
    """j9 ranks before j3 when sorted by executive_metric."""
    gs = GridSearch(cost_params=cost_params, n_runs=300, seed=42)
    results = gs.evaluate_all(financial_agent_configs)
    ranked = rank_by_executive_metric(results)
    ids = [r.config.config_id for r in ranked]
    assert ids.index("j9") < ids.index("j3")


def test_executive_metric_zero_acr() -> None:
    """executive_metric returns inf when acr == 0."""
    assert executive_metric(1.0, 0.0) == float("inf")


def test_executive_metric_normal() -> None:
    """executive_metric returns tco / acr for positive acr."""
    assert executive_metric(2.0, 0.5) == pytest.approx(4.0)
