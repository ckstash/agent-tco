from __future__ import annotations

import math

import pytest

from agent_tco.model import WorkflowConfig
from agent_tco.sensitivity import (
    critical_step_sensitivity,
    marginal_acr_gain,
    sample_complexity,
)


def test_marginal_acr_gain_formula(config_j3: WorkflowConfig) -> None:
    """Step 2 (q=0.78) marginal gain equals product of other four steps' acr values."""
    gain = marginal_acr_gain(config_j3, step_index=2)
    expected = 0.92 ** 4  # four steps each with acr=0.92
    assert gain == pytest.approx(expected, rel=1e-6)


def test_critical_step_is_lowest_q(config_j3: WorkflowConfig) -> None:
    """Step 2 (lowest acr=0.78) is the first entry in critical_step_sensitivity."""
    ranking = critical_step_sensitivity(config_j3)
    assert ranking[0][0] == 2


def test_critical_step_sensitivity_length(config_j3: WorkflowConfig) -> None:
    """critical_step_sensitivity returns one entry per step."""
    ranking = critical_step_sensitivity(config_j3)
    assert len(ranking) == config_j3.n_steps


def test_critical_step_sensitivity_descending(config_j3: WorkflowConfig) -> None:
    """Gains are sorted in descending order."""
    ranking = critical_step_sensitivity(config_j3)
    gains = [g for _, g in ranking]
    assert gains == sorted(gains, reverse=True)


def test_marginal_acr_gain_invalid_step(config_j3: WorkflowConfig) -> None:
    """ValueError is raised for a step_index not present in the config."""
    with pytest.raises(ValueError, match="step_index"):
        marginal_acr_gain(config_j3, step_index=99)


def test_sample_complexity_positive() -> None:
    """sample_complexity returns a positive integer."""
    n = sample_complexity(c_max=0.10, epsilon=0.005)
    assert isinstance(n, int)
    assert n > 0


def test_sample_complexity_increases_with_precision() -> None:
    """Smaller epsilon requires larger N."""
    n_coarse = sample_complexity(c_max=1.0, epsilon=0.10)
    n_fine = sample_complexity(c_max=1.0, epsilon=0.01)
    assert n_fine > n_coarse


def test_sample_complexity_increases_with_c_max() -> None:
    """Larger c_max requires larger N."""
    n_small = sample_complexity(c_max=0.10, epsilon=0.01)
    n_large = sample_complexity(c_max=1.00, epsilon=0.01)
    assert n_large > n_small


@pytest.mark.parametrize("bad_val,param", [
    (-1.0, "epsilon"),
    (0.0, "epsilon"),
    (-0.5, "c_max"),
    (0.0, "c_max"),
])
def test_sample_complexity_invalid_inputs(bad_val: float, param: str) -> None:
    """ValueError is raised for non-positive epsilon or c_max."""
    if param == "epsilon":
        with pytest.raises(ValueError):
            sample_complexity(c_max=1.0, epsilon=bad_val)
    else:
        with pytest.raises(ValueError):
            sample_complexity(c_max=bad_val, epsilon=0.01)
