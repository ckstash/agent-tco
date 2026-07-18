from __future__ import annotations

import pytest

from agent_tco.model import CostParams, Scenario, WorkflowConfig


def test_scenario_enum_values() -> None:
    """All six Scenario members have the correct string values i1–i6."""
    assert Scenario.HAPPY_PATH.value == "i1"
    assert Scenario.RETRY_RESOLVED.value == "i2"
    assert Scenario.HITL_SUCCESS.value == "i3"
    assert Scenario.HITL_PARTIAL.value == "i4"
    assert Scenario.COMPOUND_RECOVERY.value == "i5"
    assert Scenario.ABORTED.value == "i6"


def test_cost_params_hitl_cost(cost_params: CostParams) -> None:
    """hitl_cost_per_event equals labor_rate * review_time_hours."""
    assert cost_params.hitl_cost_per_event == pytest.approx(15.0)


def test_workflow_config_n_steps(config_j3: WorkflowConfig) -> None:
    """n_steps equals the number of StepConfig entries."""
    assert config_j3.n_steps == 5
