from __future__ import annotations

from unittest.mock import Mock

import pytest

from agent_tco.model import CostParams, StepConfig, WorkflowConfig
from agent_tco.pricing import TIER_A, TIER_B, TIER_C


COST_PARAMS = CostParams(
    labor_rate=60.0,
    review_time_hours=0.25,
    manual_process_cost=50.0,
)


def _make_step(index: int, tier, acr: float, hitl_prob: float) -> StepConfig:
    return StepConfig(
        step_index=index,
        model_tier=tier.tier_id,
        acr=acr,
        hitl_prob=hitl_prob,
        inference_cost=tier.cost_per_1k_tokens * 0.5,
    )


# Paper's j3: all Tier C
J3_STEPS = [
    _make_step(1, TIER_C, 0.92, 0.05),
    _make_step(2, TIER_C, 0.78, 0.05),
    _make_step(3, TIER_C, 0.92, 0.05),
    _make_step(4, TIER_C, 0.92, 0.20),
    _make_step(5, TIER_C, 0.92, 0.05),
]

# Paper's j9: Tier B at step 2, Tier C elsewhere
J9_STEPS = [
    _make_step(1, TIER_C, 0.92, 0.05),
    _make_step(2, TIER_B, 0.88, 0.08),
    _make_step(3, TIER_C, 0.92, 0.05),
    _make_step(4, TIER_C, 0.92, 0.20),
    _make_step(5, TIER_C, 0.92, 0.05),
]


@pytest.fixture
def cost_params() -> CostParams:
    """Return the financial agent cost parameters from the paper."""
    return COST_PARAMS


@pytest.fixture
def config_j3() -> WorkflowConfig:
    """Return paper configuration j3: all Tier C."""
    return WorkflowConfig("j3", list(J3_STEPS))


@pytest.fixture
def config_j9() -> WorkflowConfig:
    """Return paper configuration j9: Tier B at step 2, Tier C elsewhere."""
    return WorkflowConfig("j9", list(J9_STEPS))


@pytest.fixture
def financial_agent_configs() -> list[WorkflowConfig]:
    """Return a representative subset of the paper's 12 configurations."""
    j1 = WorkflowConfig("j1", [
        _make_step(k, TIER_A, 0.96 if k == 2 else 0.98, 0.03 if k == 4 else 0.01)
        for k in range(1, 6)
    ])
    j2 = WorkflowConfig("j2", [
        _make_step(k, TIER_B, 0.88 if k == 2 else 0.95, 0.08 if k == 4 else 0.02)
        for k in range(1, 6)
    ])
    j3 = WorkflowConfig("j3", list(J3_STEPS))
    j9 = WorkflowConfig("j9", list(J9_STEPS))
    return [j1, j2, j3, j9]


@pytest.fixture
def mock_step_executor() -> Mock:
    """Return a mock StepExecutor: (StepConfig, dict) -> (completed, hitl_triggered).

    Default: always completes, never triggers HITL.
    Override with mock_step_executor.return_value = (False, False) etc.
    """
    return Mock(return_value=(True, False))
