"""Reproducible synthetic inputs for the companion paper.

The profiles in this module reproduce the worked examples in *Beyond Point
Solutions: Cost-Quality Optimization of Multi-Model Agentic Workflows*.  They
are synthetic verification inputs, not estimates for a production deployment.
"""
from __future__ import annotations

from itertools import product

from agent_tco.exact import evaluate_exact
from agent_tco.grid import GridResult
from agent_tco.model import CostParams, StepConfig, WorkflowConfig
from agent_tco.topology import TaskNode, WorkflowBlock
from agent_tco.topology_search import (
    SearchReport,
    exhaustive_topology_search,
    guided_topology_search,
)


PAPER_TIERS = ("A", "B", "C")

_CHAIN_Q = {
    "A": (0.96, 0.94, 0.97, 0.95, 0.96),
    "B": (0.92, 0.86, 0.95, 0.90, 0.92),
    "C": (0.86, 0.70, 0.91, 0.80, 0.85),
}
_CHAIN_H = {
    "A": (0.010, 0.010, 0.005, 0.015, 0.010),
    "B": (0.020, 0.020, 0.010, 0.030, 0.020),
    "C": (0.030, 0.040, 0.010, 0.100, 0.030),
}
_CHAIN_P = {
    "A": (0.002, 0.002, 0.001, 0.003, 0.002),
    "B": (0.004, 0.004, 0.002, 0.006, 0.004),
    "C": (0.008, 0.008, 0.004, 0.015, 0.008),
}
_INFERENCE_COST = {"A": 0.50, "B": 0.08, "C": 0.003}

PAPER_CHAIN_COSTS = CostParams(
    labor_rate=60.0,
    review_time_hours=0.25,
    manual_process_cost=30.0,
)


def build_paper_chain_config(tiers: str | tuple[str, ...]) -> WorkflowConfig:
    """Build one five-step financial-close assignment from its tier string.

    The validation-step assisted-outcome profile depends on the classification
    tier, matching the explicit inter-step link declared in the manuscript.
    """
    combo = tuple(tiers)
    if len(combo) != 5 or any(tier not in PAPER_TIERS for tier in combo):
        raise ValueError("tiers must contain exactly five values from 'A', 'B', and 'C'")

    steps = []
    for index, tier in enumerate(combo):
        hitl_prob = _CHAIN_H[tier][index]
        partial_prob = _CHAIN_P[tier][index]
        if index == 3:
            classification_tier = combo[1]
            hitl_prob = {"A": 0.015, "B": 0.030, "C": 0.045}[classification_tier]
            partial_prob = {"A": 0.003, "B": 0.004, "C": 0.005}[classification_tier]
        steps.append(
            StepConfig(
                step_index=index + 1,
                model_tier=tier,
                q=_CHAIN_Q[tier][index],
                hitl_prob=hitl_prob,
                partial_prob=partial_prob,
                inference_cost=_INFERENCE_COST[tier],
            )
        )
    return WorkflowConfig("".join(combo), steps)


def paper_chain_configurations() -> list[WorkflowConfig]:
    """Return all 3^5 assignments in deterministic Cartesian-product order."""
    return [build_paper_chain_config(combo) for combo in product(PAPER_TIERS, repeat=5)]


def evaluate_paper_chain() -> list[GridResult]:
    """Exactly evaluate all 243 assignments in the paper's chain example."""
    return [evaluate_exact(config, PAPER_CHAIN_COSTS) for config in paper_chain_configurations()]


def paper_chain_screened_ids() -> set[str]:
    """Return the 13 homogeneous and single-upgrade assignments in the paper."""
    identifiers = {"AAAAA", "BBBBB", "CCCCC"}
    for index in range(5):
        for tier in ("A", "B"):
            combo = ["C"] * 5
            combo[index] = tier
            identifiers.add("".join(combo))
    return identifiers


PAPER_STRUCTURED_NODE_IDS = (
    "primary_extract",
    "fallback_extract",
    "reconcile",
    "intercompany",
    "anomaly",
    "classify",
    "validate",
    "report",
)
_STRUCTURED_Q = {
    "A": (0.97, 0.95, 0.96, 0.94, 0.92, 0.95, 0.96, 0.97),
    "B": (0.93, 0.90, 0.91, 0.87, 0.84, 0.88, 0.90, 0.92),
    "C": (0.86, 0.80, 0.83, 0.74, 0.69, 0.76, 0.79, 0.84),
}
_STRUCTURED_H = {
    "A": (0.008, 0.010, 0.010, 0.012, 0.015, 0.010, 0.010, 0.008),
    "B": (0.018, 0.020, 0.022, 0.028, 0.035, 0.025, 0.022, 0.018),
    "C": (0.035, 0.040, 0.050, 0.075, 0.090, 0.060, 0.055, 0.040),
}
_STRUCTURED_P = {
    "A": (0.001, 0.001, 0.002, 0.002, 0.003, 0.002, 0.002, 0.001),
    "B": (0.003, 0.003, 0.004, 0.005, 0.006, 0.004, 0.004, 0.003),
    "C": (0.007, 0.008, 0.010, 0.015, 0.018, 0.012, 0.010, 0.008),
}

PAPER_STRUCTURED_COSTS = CostParams(
    labor_rate=60.0,
    review_time_hours=5.0 / 60.0,
    manual_process_cost=10.0,
)
PAPER_GUIDED_BUDGET = 500


def build_paper_structured_topology() -> WorkflowBlock:
    """Build the paper's eight-touchpoint structured close-control workflow."""
    tasks = {node_id: TaskNode(node_id) for node_id in PAPER_STRUCTURED_NODE_IDS}
    return WorkflowBlock.series(
        WorkflowBlock.or_fallback(tasks["primary_extract"], tasks["fallback_extract"]),
        WorkflowBlock.and_eager(
            tasks["reconcile"], tasks["intercompany"], tasks["anomaly"]
        ),
        WorkflowBlock.retry_bounded(tasks["classify"], max_attempts=2),
        tasks["validate"],
        tasks["report"],
    )


def paper_structured_options() -> dict[str, tuple[StepConfig, ...]]:
    """Return the three model options for each structured-example task."""
    return {
        node_id: tuple(
            StepConfig(
                step_index=index + 1,
                model_tier=tier,
                q=_STRUCTURED_Q[tier][index],
                hitl_prob=_STRUCTURED_H[tier][index],
                partial_prob=_STRUCTURED_P[tier][index],
                inference_cost=_INFERENCE_COST[tier],
            )
            for tier in PAPER_TIERS
        )
        for index, node_id in enumerate(PAPER_STRUCTURED_NODE_IDS)
    }


def evaluate_paper_structured_exhaustive() -> SearchReport:
    """Exactly evaluate all 6,561 structured-example assignments."""
    return exhaustive_topology_search(
        build_paper_structured_topology(),
        paper_structured_options(),
        PAPER_STRUCTURED_COSTS,
    )


def evaluate_paper_structured_guided(
    max_evaluations: int = PAPER_GUIDED_BUDGET,
) -> SearchReport:
    """Run the paper's deterministic budgeted structured-assignment search."""
    return guided_topology_search(
        build_paper_structured_topology(),
        paper_structured_options(),
        PAPER_STRUCTURED_COSTS,
        max_evaluations=max_evaluations,
    )