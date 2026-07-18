from __future__ import annotations

from dataclasses import dataclass

from agent_tco.model import StepConfig


@dataclass
class TierCost:
    """Per-1K-output-token cost for one model tier."""

    tier_id: str
    cost_per_1k_tokens: float  # dollars


# Built-in tier cost defaults. Override with your own TierCost if pricing has changed.
TIER_A = TierCost("A", 0.003)    # frontier model (e.g. GPT-4o / Claude Sonnet class)
TIER_B = TierCost("B", 0.0008)   # mid-range model (e.g. GPT-4o Mini / Claude Haiku class)
TIER_C = TierCost("C", 0.00015)  # lightweight model


def inference_cost(output_tokens: int, tier: TierCost) -> float:
    """Return the inference cost in dollars for output_tokens at tier pricing."""
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
