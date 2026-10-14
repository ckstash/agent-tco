from __future__ import annotations

from agent_tco.config_space import ConfigSpaceBuilder, TierOption
from agent_tco.executor import StepExecutor
from agent_tco.experiment import (
    ExperimentRecord,
    ExperimentReport,
    AsyncProductionExecutor,
    AsyncProductionExperimentRunner,
    MeasuredConfigResult,
    MeasuredRun,
    ProductionExecutor,
    ProductionExperimentRunner,
    aggregate_measured,
    measured_pareto_frontier,
)
from agent_tco.exact import evaluate_exact
from agent_tco.grid import (
    GridResult,
    GridSearch,
    ParallelGridSearch,
    ScenarioStats,
    recompute_tco,
)
from agent_tco.model import CostParams, Scenario, SimulationRun, StepConfig, StepOutcome, WorkflowConfig
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
from agent_tco.dag import (
    CancellationPolicy,
    DAGEdge,
    DAGNode,
    DAGRun,
    DAGSimulationResult,
    DAGSimulator,
    FailureDomain,
    JoinPolicy,
    NodeExecution,
    NodeStatus,
    OutgoingPolicy,
    SuccessPolicy,
    TaskProfile,
    WorkflowGraph,
)
from agent_tco.dag_search import (
    DAGSearchReport,
    budgeted_dag_search,
    dag_pareto_frontier,
)
from agent_tco.hybrid import evaluate_hybrid
from agent_tco.hypotheses import run_stylized_hypothesis_tests
from agent_tco.dag_io import graph_from_dict, load_graph, load_profiles, profiles_from_dict
from agent_tco.plot import plot_measured_pareto_frontier
from agent_tco.paper import (
    PAPER_CHAIN_COSTS,
    PAPER_GUIDED_BUDGET,
    PAPER_STRUCTURED_COSTS,
    PAPER_STRUCTURED_NODE_IDS,
    PAPER_TIERS,
    build_paper_chain_config,
    build_paper_structured_topology,
    evaluate_paper_chain,
    evaluate_paper_structured_exhaustive,
    evaluate_paper_structured_guided,
    paper_chain_configurations,
    paper_chain_screened_ids,
    paper_structured_options,
)
from agent_tco.topology import (
    BlockType,
    TaskNode,
    TopologyResult,
    WorkflowBlock,
    birnbaum_importance,
    collect_task_ids,
    evaluate_topology,
    topology_acr,
)
from agent_tco.topology_search import (
    SearchReport,
    exhaustive_topology_search,
    guided_topology_search,
    topology_pareto_frontier,
)

__all__ = [
    # model
    "Scenario",
    "StepConfig",
    "StepOutcome",
    "WorkflowConfig",
    "CostParams",
    "SimulationRun",
    # executor
    "StepExecutor",
    "evaluate_exact",
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
    # structured topology
    "BlockType",
    "TaskNode",
    "WorkflowBlock",
    "TopologyResult",
    "collect_task_ids",
    "evaluate_topology",
    "topology_acr",
    "birnbaum_importance",
    # topology search
    "SearchReport",
    "topology_pareto_frontier",
    "exhaustive_topology_search",
    "guided_topology_search",
    # profile-based DAG simulation
    "JoinPolicy",
    "OutgoingPolicy",
    "SuccessPolicy",
    "CancellationPolicy",
    "NodeStatus",
    "TaskProfile",
    "FailureDomain",
    "DAGNode",
    "DAGEdge",
    "WorkflowGraph",
    "NodeExecution",
    "DAGRun",
    "DAGSimulationResult",
    "DAGSimulator",
    "DAGSearchReport",
    "dag_pareto_frontier",
    "budgeted_dag_search",
    "evaluate_hybrid",
    "run_stylized_hypothesis_tests",
    "graph_from_dict",
    "load_graph",
    "profiles_from_dict",
    "load_profiles",
    # measured production-like experiments
    "MeasuredRun",
    "ProductionExecutor",
    "ExperimentRecord",
    "MeasuredConfigResult",
    "ExperimentReport",
    "ProductionExperimentRunner",
    "AsyncProductionExecutor",
    "AsyncProductionExperimentRunner",
    "aggregate_measured",
    "measured_pareto_frontier",
    "plot_measured_pareto_frontier",
    # companion-paper reproduction inputs
    "PAPER_TIERS",
    "PAPER_CHAIN_COSTS",
    "PAPER_STRUCTURED_NODE_IDS",
    "PAPER_STRUCTURED_COSTS",
    "PAPER_GUIDED_BUDGET",
    "build_paper_chain_config",
    "paper_chain_configurations",
    "evaluate_paper_chain",
    "paper_chain_screened_ids",
    "build_paper_structured_topology",
    "paper_structured_options",
    "evaluate_paper_structured_exhaustive",
    "evaluate_paper_structured_guided",
]
