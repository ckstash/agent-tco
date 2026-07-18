from __future__ import annotations

import pytest

from agent_tco.pricing import (
    TIER_A,
    TIER_B,
    TIER_C,
    TierCost,
    inference_cost,
    step_config_from_tier,
)


def test_tier_defaults() -> None:
    """Built-in tier costs match the paper's July 2026 pricing."""
    assert TIER_A.cost_per_1k_tokens == pytest.approx(0.003)
    assert TIER_B.cost_per_1k_tokens == pytest.approx(0.0008)
    assert TIER_C.cost_per_1k_tokens == pytest.approx(0.00015)


def test_inference_cost_tier_a() -> None:
    """inference_cost(1000, TIER_A) equals TIER_A.cost_per_1k_tokens."""
    assert inference_cost(1000, TIER_A) == pytest.approx(0.003)


def test_inference_cost_proportional() -> None:
    """inference_cost scales linearly with token count."""
    assert inference_cost(500, TIER_A) == pytest.approx(0.0015)
    assert inference_cost(2000, TIER_A) == pytest.approx(0.006)


def test_step_config_from_tier_cost_field() -> None:
    """step_config_from_tier sets inference_cost = inference_cost(avg_output_tokens, tier)."""
    step = step_config_from_tier(
        step_index=1, tier=TIER_B, acr=0.92, hitl_prob=0.05, avg_output_tokens=500
    )
    expected = inference_cost(500, TIER_B)
    assert step.inference_cost == pytest.approx(expected)


def test_step_config_from_tier_fields() -> None:
    """step_config_from_tier copies step_index, tier_id, acr, hitl_prob correctly."""
    step = step_config_from_tier(
        step_index=3, tier=TIER_C, acr=0.88, hitl_prob=0.10
    )
    assert step.step_index == 3
    assert step.model_tier == "C"
    assert step.acr == pytest.approx(0.88)
    assert step.hitl_prob == pytest.approx(0.10)
