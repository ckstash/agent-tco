"""Persistence utilities for GridResult objects.

The key practical advantage of agent-tco is that scenario probability estimates
are independent of cost parameters. Save results after a grid search, then
recompute the Pareto frontier under different pricing without re-simulating.

Example::

    from agent_tco.persist import save_results, load_results
    from agent_tco.grid import recompute_tco
    from agent_tco.pareto import pareto_frontier

    # Save after grid search
    save_results(results, "run_2026_08_01.json")

    # Later, with updated cost params
    results = load_results("run_2026_08_01.json")
    frontier = pareto_frontier(results)
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
            "acr": s.acr,
            "hitl_prob": s.hitl_prob,
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
        "pareto_dominated": r.pareto_dominated,
    }


def _result_from_dict(d: dict) -> GridResult:
    steps = [
        StepConfig(
            step_index=s["step_index"],
            model_tier=s["model_tier"],
            acr=s["acr"],
            hitl_prob=s["hitl_prob"],
            inference_cost=s["inference_cost"],
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
        )
    exec_metric = d["executive_metric"] if d["executive_metric"] is not None else float("inf")
    return GridResult(
        config=config,
        n_runs=d["n_runs"],
        scenario_stats=scenario_stats,
        acr=d["acr"],
        expected_tco=d["expected_tco"],
        executive_metric=exec_metric,
        pareto_dominated=d["pareto_dominated"],
    )


def save_results(results: list[GridResult], path: str | Path) -> None:
    """Save a list of GridResults to a JSON file.

    The saved file contains scenario probabilities and costs. Because these
    are independent of cost parameters, you can reload and recompute the
    Pareto frontier under different pricing without re-running simulations.
    """
    data = [_result_to_dict(r) for r in results]
    Path(path).write_text(json.dumps(data, indent=2))


def load_results(path: str | Path) -> list[GridResult]:
    """Load a list of GridResults from a JSON file previously saved by save_results()."""
    data = json.loads(Path(path).read_text())
    return [_result_from_dict(d) for d in data]
