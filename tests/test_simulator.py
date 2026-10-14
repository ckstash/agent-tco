from __future__ import annotations

from unittest.mock import Mock

import pytest

from agent_tco.model import CostParams, Scenario, StepConfig, StepOutcome, WorkflowConfig
from agent_tco.pricing import TIER_C
from agent_tco.simulator import WorkflowSimulator


def _always_succeed_config(cost: float = 0.001) -> WorkflowConfig:
    """Return a 2-step config with q=1.0, hitl_prob=0 — always happy path."""
    steps = [
        StepConfig(step_index=1, model_tier="C", q=1.0, hitl_prob=0.0, inference_cost=cost),
        StepConfig(step_index=2, model_tier="C", q=1.0, hitl_prob=0.0, inference_cost=cost),
    ]
    return WorkflowConfig("always_ok", steps)


def _always_fail_config(cost: float = 0.001) -> WorkflowConfig:
    """Return a 2-step config with q=0.0 — always aborted."""
    steps = [
        StepConfig(step_index=1, model_tier="C", q=0.0, hitl_prob=0.0, inference_cost=cost),
        StepConfig(step_index=2, model_tier="C", q=0.0, hitl_prob=0.0, inference_cost=cost),
    ]
    return WorkflowConfig("always_fail", steps)


def _hitl_config(cost: float = 0.001) -> WorkflowConfig:
    """Return a one-step configuration with HITL-assisted success on every run."""
    steps = [
        StepConfig(step_index=1, model_tier="C", q=0.0, hitl_prob=1.0, inference_cost=cost),
    ]
    return WorkflowConfig("hitl", steps)


def test_happy_path_scenario(cost_params: CostParams) -> None:
    """All steps succeed, no HITL → HAPPY_PATH."""
    sim = WorkflowSimulator(cost_params=cost_params, seed=0)
    run = sim.simulate_run(_always_succeed_config())
    assert run.scenario == Scenario.HAPPY_PATH
    assert run.hitl_triggered is False
    assert run.retry_count == 0


def test_aborted_scenario(cost_params: CostParams) -> None:
    """Step fails on both attempts → ABORTED and cost includes manual_process_cost."""
    sim = WorkflowSimulator(cost_params=cost_params, seed=0)
    run = sim.simulate_run(_always_fail_config(cost=0.001))
    assert run.scenario == Scenario.ABORTED
    assert run.total_cost >= cost_params.manual_process_cost


def test_hitl_success_scenario(cost_params: CostParams) -> None:
    """HITL triggered, step completes, no retry → HITL_SUCCESS."""
    sim = WorkflowSimulator(cost_params=cost_params, seed=0)
    run = sim.simulate_run(_hitl_config())
    assert run.scenario == Scenario.HITL_SUCCESS
    assert run.hitl_triggered is True
    assert run.total_cost >= cost_params.hitl_cost_per_event


def test_reproducibility(cost_params: CostParams, mixed_config: WorkflowConfig) -> None:
    """Same seed + run_index produces identical SimulationRun."""
    sim = WorkflowSimulator(cost_params=cost_params, seed=42)
    r1 = sim.simulate_run(mixed_config, run_index=7)
    r2 = sim.simulate_run(mixed_config, run_index=7)
    assert r1.scenario == r2.scenario
    assert r1.total_cost == r2.total_cost
    assert r1.retry_count == r2.retry_count


@pytest.mark.parametrize(
    "retry_count,hitl_triggered,terminal,expected",
    [
        (0, False, "success", Scenario.HAPPY_PATH),
        (1, False, "success", Scenario.RETRY_RESOLVED),
        (0, True, "success", Scenario.HITL_SUCCESS),
        (0, True, "partial", Scenario.HITL_PARTIAL),
        (1, True, "success", Scenario.COMPOUND_RECOVERY),
        (0, False, "aborted", Scenario.ABORTED),
    ],
)
def test_classify_scenario_exhaustive(
    retry_count: int,
    hitl_triggered: bool,
    terminal: str,
    expected: Scenario,
    cost_params: CostParams,
) -> None:
    """classify_scenario covers all six scenario classes."""
    sim = WorkflowSimulator(cost_params=cost_params)
    assert sim.classify_scenario(retry_count, hitl_triggered, terminal) == expected


def test_executor_called_per_step(
    mock_step_executor: Mock,
    mixed_config: WorkflowConfig,
    cost_params: CostParams,
) -> None:
    """With executor provided, it is called exactly n_steps times for a completing run."""
    mock_step_executor.return_value = (True, False)
    sim = WorkflowSimulator(cost_params=cost_params, executor=mock_step_executor)
    sim.simulate_run(mixed_config)
    assert mock_step_executor.call_count == mixed_config.n_steps


def test_executor_overrides_bernoulli(
    mock_step_executor: Mock,
    mixed_config: WorkflowConfig,
    cost_params: CostParams,
) -> None:
    """Executor returning (False, False) aborts every run regardless of step.q."""
    mock_step_executor.return_value = (False, False)
    sim = WorkflowSimulator(cost_params=cost_params, executor=mock_step_executor)
    for i in range(5):
        run = sim.simulate_run(mixed_config, run_index=i)
        assert run.scenario == Scenario.ABORTED


def test_executor_hitl_flag(
    mock_step_executor: Mock,
    mixed_config: WorkflowConfig,
    cost_params: CostParams,
) -> None:
    """Executor returning (True, True) sets hitl_triggered and includes HITL cost."""
    mock_step_executor.return_value = (True, True)
    sim = WorkflowSimulator(cost_params=cost_params, executor=mock_step_executor)
    run = sim.simulate_run(mixed_config)
    assert run.hitl_triggered is True
    assert run.total_cost >= cost_params.hitl_cost_per_event


def test_hitl_success_continues_to_downstream_step(cost_params: CostParams) -> None:
    executor = Mock(side_effect=[StepOutcome.HITL_SUCCESS, StepOutcome.AUTONOMOUS])
    config = WorkflowConfig(
        "assisted_then_autonomous",
        [
            StepConfig(1, "C", q=0.0, hitl_prob=1.0, inference_cost=0.001),
            StepConfig(2, "C", q=1.0, hitl_prob=0.0, inference_cost=0.002),
        ],
    )
    run = WorkflowSimulator(cost_params=cost_params, executor=executor).simulate_run(config)

    assert executor.call_count == 2
    assert run.scenario == Scenario.HITL_SUCCESS
    assert run.total_cost == pytest.approx(cost_params.hitl_cost_per_event + 0.003)


def test_partial_scenario_reachable(cost_params: CostParams) -> None:
    """A categorical partial outcome reaches the HITL_PARTIAL scenario."""
    executor = Mock(return_value=StepOutcome.HITL_PARTIAL)
    config = WorkflowConfig("partial", [StepConfig(1, "C", 0.0, 0.0, 0.001, 1.0)])
    run = WorkflowSimulator(cost_params=cost_params, executor=executor).simulate_run(config)
    assert run.scenario == Scenario.HITL_PARTIAL


def test_compound_recovery_reachable(cost_params: CostParams) -> None:
    """A retry followed by review reaches the COMPOUND_RECOVERY scenario."""
    executor = Mock(side_effect=[StepOutcome.INCOMPLETE, StepOutcome.HITL_SUCCESS])
    config = WorkflowConfig("compound", [StepConfig(1, "C", 0.0, 0.5, 0.001)])
    run = WorkflowSimulator(cost_params=cost_params, executor=executor).simulate_run(config)
    assert run.scenario == Scenario.COMPOUND_RECOVERY


def test_retry_resolved_reachable(cost_params: CostParams) -> None:
    """A non-completion followed by autonomous completion reaches RETRY_RESOLVED."""
    executor = Mock(side_effect=[StepOutcome.INCOMPLETE, StepOutcome.AUTONOMOUS])
    config = WorkflowConfig("retry", [StepConfig(1, "C", 0.5, 0.0, 0.001)])
    run = WorkflowSimulator(cost_params=cost_params, executor=executor).simulate_run(config)
    assert run.scenario == Scenario.RETRY_RESOLVED
    assert run.retry_count == 1


def test_categorical_probabilities_must_be_disjoint() -> None:
    """Overlapping categorical probabilities are rejected."""
    with pytest.raises(ValueError, match="must be <= 1"):
        StepConfig(1, "C", q=0.9, hitl_prob=0.2, inference_cost=0.001)
