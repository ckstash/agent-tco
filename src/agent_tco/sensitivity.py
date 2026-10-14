from __future__ import annotations

import math

from agent_tco.model import WorkflowConfig


def marginal_acr_gain(config: WorkflowConfig, step_index: int) -> float:
    """Return the derivative of chain ACR with respect to one per-attempt rate.

    Let u_k = 1-q_k-h_k-p_k be the per-attempt non-completion probability.
    With one retry, a step passes autonomously with probability
    a_k = q_k(1+u_k). Chain ACR is the product of the a_k values, so
    dACR/dq_k = (2-2q_k-h_k-p_k) product_{j != k} a_j when h_k and p_k
    are held fixed.
    """
    indices = [s.step_index for s in config.steps]
    if step_index not in indices:
        raise ValueError(
            f"step_index {step_index} not found in config steps {indices}"
        )
    target = next(s for s in config.steps if s.step_index == step_index)
    result = 2.0 - 2.0 * target.q - target.hitl_prob - target.partial_prob
    for step in config.steps:
        if step.step_index != step_index:
            result *= step.autonomous_pass_prob
    return result


def critical_step_sensitivity(config: WorkflowConfig) -> list[tuple[int, float]]:
    """Return (step_index, marginal_acr_gain) pairs sorted descending by gain.

    The first entry is the highest local ACR-sensitivity step. This ranking is a
    screening heuristic because available upgrades have different costs and
    finite probability increments.
    """
    gains = [
        (step.step_index, marginal_acr_gain(config, step.step_index))
        for step in config.steps
    ]
    return sorted(gains, key=lambda x: x[1], reverse=True)


def sensitivity_report(
    config: WorkflowConfig,
    step_labels: dict[int, str] | None = None,
) -> str:
    """Return a formatted sensitivity ranking table as a plain-text string.

    Args:
        config: the baseline WorkflowConfig (typically all-Tier-C).
        step_labels: optional mapping of step_index -> human-readable label.

    Example output:
        Rank  Step  Label               Sensitivity
        ------------------------------------------------
           1     2  Classify accounts       0.4454  ← highest sensitivity
           2     4  Validate totals         0.2690
           3     5  Generate report         0.1960
           4     1  Extract trial balance   0.1806
           5     3  Compute totals          0.1199
    """
    ranking = critical_step_sensitivity(config)
    lines = [
        f"{'Rank':>4}  {'Step':>4}  {'Label':<22}  {'Sensitivity':>11}",
        "-" * 50,
    ]
    for rank, (idx, gain) in enumerate(ranking, 1):
        label = (step_labels or {}).get(idx, f"step_{idx}")
        marker = "  ← highest sensitivity" if rank == 1 else ""
        lines.append(f"{rank:>4}  {idx:>4}  {label:<22}  {gain:>11.4f}{marker}")
    return "\n".join(lines)


def sample_complexity(
    c_max: float,
    epsilon: float,
    delta: float = 0.05,
) -> int:
    """Return the minimum N for E[TCO] estimation within epsilon at confidence 1-delta.

    Uses the direct Hoeffding bound for the sample mean:
    N >= ceil(c_max^2 * log(2 / delta) / (2 * epsilon^2))
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive; got {epsilon}")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1); got {delta}")
    if c_max <= 0:
        raise ValueError(f"c_max must be positive; got {c_max}")
    n = (c_max ** 2 * math.log(2 / delta)) / (2 * epsilon ** 2)
    return math.ceil(n)
