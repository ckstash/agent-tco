from __future__ import annotations

from agent_tco.config_space import ConfigSpaceBuilder, TierOption
from agent_tco.executor import StepExecutor
from agent_tco.grid import (
    GridResult,
    GridSearch,
    ParallelGridSearch,
    ScenarioStats,
    recompute_tco,
)
from agent_tco.model import CostParams, Scenario, SimulationRun, StepConfig, WorkflowConfig
from agent_tco.pareto import (
    dominates,
    executive_metric,
    pareto_frontier,
    rank_by_executive_metric,
)
from agent_tco.persist import load_results, save_results
from agent_tco.pricing import (
    TIER_A,
    TIER_B,
    TIER_C,
    TIER_DEFAULT_A,
    TIER_DEFAULT_B,
    TIER_DEFAULT_C,
    TierCost,
    inference_cost,
    step_config_from_tier,
)
from agent_tco.sensitivity import (
    critical_step_sensitivity,
    marginal_acr_gain,
    sample_complexity,
    sensitivity_report,
)
from agent_tco.simulator import WorkflowSimulator

__all__ = [
    # model
    "Scenario",
    "StepConfig",
    "WorkflowConfig",
    "CostParams",
    "SimulationRun",
    # executor
    "StepExecutor",
    # simulator
    "WorkflowSimulator",
    # grid
    "ScenarioStats",
    "GridResult",
    "GridSearch",
    "ParallelGridSearch",
    "recompute_tco",
    # pareto
    "dominates",
    "pareto_frontier",
    "executive_metric",
    "rank_by_executive_metric",
    # sensitivity
    "marginal_acr_gain",
    "critical_step_sensitivity",
    "sensitivity_report",
    "sample_complexity",
    # pricing
    "TierCost",
    "TIER_A",
    "TIER_B",
    "TIER_C",
    "TIER_DEFAULT_A",
    "TIER_DEFAULT_B",
    "TIER_DEFAULT_C",
    "inference_cost",
    "step_config_from_tier",
    # persist
    "save_results",
    "load_results",
    # config_space
    "ConfigSpaceBuilder",
    "TierOption",
]
