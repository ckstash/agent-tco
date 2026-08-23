"""Tier cost definitions for the agent-tco cost model.

The TIER_DEFAULT_* constants (also aliased as TIER_A/B/C) are illustrative
starting-point values only. Actual API pricing changes frequently and varies
by provider, region, and contract. Always supply your current pricing by
constructing TierCost objects directly.

Example::

    tier_a = TierCost("A", cost_per_1k_tokens=0.015)   # your actual rate
    tier_c = TierCost("C", cost_per_1k_tokens=0.0002)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from agent_tco.model import StepConfig


@dataclass(frozen=True)
class TierCost:
    """Per-1K-output-token inference cost for one model tier."""

    tier_id: str
    cost_per_1k_tokens: float  # dollars; always supply your actual rate


# Illustrative defaults. NOT current API pricing.
# Override with TierCost(tier_id, cost_per_1k_tokens=<your rate>) for any real deployment.
TIER_DEFAULT_A = TierCost("A", 0.003)
TIER_DEFAULT_B = TierCost("B", 0.0008)
TIER_DEFAULT_C = TierCost("C", 0.00015)

# Convenience aliases (same objects)
TIER_A = TIER_DEFAULT_A
TIER_B = TIER_DEFAULT_B
TIER_C = TIER_DEFAULT_C

_DEFAULT_IDS = frozenset({id(TIER_DEFAULT_A), id(TIER_DEFAULT_B), id(TIER_DEFAULT_C)})
_WARNED = False


def _maybe_warn_defaults(tier: TierCost) -> None:
    global _WARNED
    if not _WARNED and id(tier) in _DEFAULT_IDS:
        warnings.warn(
            "Using illustrative tier cost defaults (TIER_A/B/C). "
            "These are not current API pricing. "
            "Pass TierCost(tier_id, cost_per_1k_tokens=<your rate>) "
            "for a production analysis.",
            UserWarning,
            stacklevel=3,
        )
        _WARNED = True


def inference_cost(output_tokens: int, tier: TierCost) -> float:
    """Return the inference cost in dollars for output_tokens at tier pricing."""
    _maybe_warn_defaults(tier)
    return (output_tokens / 1000) * tier.cost_per_1k_tokens


def step_config_from_tier(
    step_index: int,
    tier: TierCost,
    acr: float,
    hitl_prob: float,
    avg_output_tokens: int = 500,
) -> StepConfig:
    """Construct a StepConfig from a TierCost and estimated average token usage."""
    return StepConfig(
        step_index=step_index,
        model_tier=tier.tier_id,
        acr=acr,
        hitl_prob=hitl_prob,
        inference_cost=inference_cost(avg_output_tokens, tier),
    )
