"""Persistence utilities for GridResult objects.

Scenario probability estimates are independent of review and fallback prices.
Save results after a grid search, then reprice those components and recompute
the Pareto frontier without re-simulating workflow behavior.

Example::

    from agent_tco.persist import save_results, load_results
    from agent_tco.grid import recompute_tco
    from agent_tco.pareto import pareto_frontier

    # Save after grid search
    save_results(results, "run_2026_08_01.json")

    # Later, with updated review or fallback prices
    results = load_results("run_2026_08_01.json")
    repriced = recompute_tco(results, updated_cost_params)
    frontier = pareto_frontier(repriced)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from agent_tco.grid import GridResult, ScenarioStats
from agent_tco.model import CostParams, Scenario, StepConfig, WorkflowConfig


def _result_to_dict(r: GridResult) -> dict:
    steps = [
        {
            "step_index": s.step_index,
            "model_tier": s.model_tier,
            "q": s.q,
            "hitl_prob": s.hitl_prob,
            "partial_prob": s.partial_prob,
            "inference_cost": s.inference_cost,
        }
        for s in r.config.steps
    ]
    scenario_stats = {
        sc.value: {
            "probability": stats.probability,
            "ci_lower": stats.ci_lower,
            "ci_upper": stats.ci_upper,
            "mean_cost": stats.mean_cost,
            "mean_review_events": stats.mean_review_events,
        }
        for sc, stats in r.scenario_stats.items()
    }
    return {
        "config_id": r.config.config_id,
        "steps": steps,
        "n_runs": r.n_runs,
        "scenario_stats": scenario_stats,
        "acr": r.acr,
        "expected_tco": r.expected_tco,
        "executive_metric": r.executive_metric if math.isfinite(r.executive_metric) else None,
        "acr_ci_lower": r.acr_ci_lower,
        "acr_ci_upper": r.acr_ci_upper,
        "tco_ci_lower": r.tco_ci_lower,
        "tco_ci_upper": r.tco_ci_upper,
        "pareto_dominated": r.pareto_dominated,
        "cost_params": (
            {
                "labor_rate": r.cost_params.labor_rate,
                "review_time_hours": r.cost_params.review_time_hours,
                "manual_process_cost": r.cost_params.manual_process_cost,
            }
            if r.cost_params is not None
            else None
        ),
    }


def _result_from_dict(d: dict) -> GridResult:
    steps = [
        StepConfig(
            step_index=s["step_index"],
            model_tier=s["model_tier"],
            q=s.get("q", s.get("acr")),
            hitl_prob=s["hitl_prob"],
            inference_cost=s["inference_cost"],
            partial_prob=s.get("partial_prob", 0.0),
        )
        for s in d["steps"]
    ]
    config = WorkflowConfig(config_id=d["config_id"], steps=steps)
    scenario_stats = {}
    for sc_val, stats in d["scenario_stats"].items():
        sc = Scenario(sc_val)
        scenario_stats[sc] = ScenarioStats(
            scenario=sc,
            probability=stats["probability"],
            ci_lower=stats["ci_lower"],
            ci_upper=stats["ci_upper"],
            mean_cost=stats["mean_cost"],
            mean_review_events=stats.get(
                "mean_review_events",
                1.0 if sc in {Scenario.HITL_SUCCESS, Scenario.HITL_PARTIAL, Scenario.COMPOUND_RECOVERY} else 0.0,
            ),
        )
    exec_metric = d["executive_metric"] if d["executive_metric"] is not None else float("inf")
    raw_cost_params = d.get("cost_params")
    cost_params = CostParams(**raw_cost_params) if raw_cost_params is not None else None
    return GridResult(
        config=config,
        n_runs=d["n_runs"],
        scenario_stats=scenario_stats,
        acr=d["acr"],
        expected_tco=d["expected_tco"],
        executive_metric=exec_metric,
        acr_ci_lower=d.get("acr_ci_lower", d["acr"]),
        acr_ci_upper=d.get("acr_ci_upper", d["acr"]),
        tco_ci_lower=d.get("tco_ci_lower", d["expected_tco"]),
        tco_ci_upper=d.get("tco_ci_upper", d["expected_tco"]),
        pareto_dominated=d["pareto_dominated"],
        cost_params=cost_params,
    )


def save_results(results: list[GridResult], path: str | Path) -> None:
    """Save a list of GridResults to a JSON file.

    New files include the original cost parameters so review and fallback
    prices can be changed later with ``recompute_tco``. Inference prices are
    part of the workflow configuration and are not changed by repricing.
    """
    data = [_result_to_dict(r) for r in results]
    Path(path).write_text(json.dumps(data, indent=2))


def load_results(path: str | Path) -> list[GridResult]:
    """Load a list of GridResults from a JSON file previously saved by save_results()."""
    data = json.loads(Path(path).read_text())
    return [_result_from_dict(d) for d in data]
