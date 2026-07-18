from __future__ import annotations

import math

from agent_tco.model import WorkflowConfig


def marginal_acr_gain(config: WorkflowConfig, step_index: int) -> float:
    """Return the marginal ACR gain from improving step step_index by one unit.

    ∂ACR/∂q_k = product of q_j for all j != step_index.
    step_index is 1-based. Uses the no-retry independence assumption.
    """
    indices = [s.step_index for s in config.steps]
    if step_index not in indices:
        raise ValueError(
            f"step_index {step_index} not found in config steps {indices}"
        )
    result = 1.0
    for step in config.steps:
        if step.step_index != step_index:
            result *= step.acr
    return result


def critical_step_sensitivity(config: WorkflowConfig) -> list[tuple[int, float]]:
    """Return (step_index, marginal_acr_gain) pairs sorted descending by gain.

    The first entry is the bottleneck step — the highest-priority upgrade target.
    """
    gains = [
        (step.step_index, marginal_acr_gain(config, step.step_index))
        for step in config.steps
    ]
    return sorted(gains, key=lambda x: x[1], reverse=True)


def sample_complexity(
    c_max: float,
    epsilon: float,
    delta: float = 0.05,
) -> int:
    """Return the minimum N for E[TCO] estimation within epsilon at confidence 1-delta.

    Uses the Hoeffding bound:
    N >= ceil(c_max^2 * log(6 / delta) / (2 * epsilon^2))
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive; got {epsilon}")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1); got {delta}")
    if c_max <= 0:
        raise ValueError(f"c_max must be positive; got {c_max}")
    n = (c_max ** 2 * math.log(6 / delta)) / (2 * epsilon ** 2)
    return math.ceil(n)
