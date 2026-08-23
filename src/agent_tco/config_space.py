"""Configuration space enumeration and targeted grid generation.

For a workflow with K steps and T tier options per step, the full configuration
space has T^K entries — 3^5 = 243 for 5 steps and 3 tiers, 5^10 ≈ 10 million
for 10 steps and 5 tiers. This module provides tools to enumerate the full
space and, more usefully, to generate the small targeted grid that the critical
step sensitivity ranking recommends simulating.

Example::

    from agent_tco.config_space import ConfigSpaceBuilder, TierOption
    from agent_tco.pricing import TierCost

    tier_a = TierCost("A", 0.50)
    tier_b = TierCost("B", 0.08)
    tier_c = TierCost("C", 0.003)

    builder = ConfigSpaceBuilder(steps=[
        TierOption(
            step_index=1, label="Extract",
            tiers=[tier_a, tier_b, tier_c],
            acr_by_tier={tier_a: 0.98, tier_b: 0.96, tier_c: 0.93},
            hitl_by_tier={tier_a: 0.01, tier_b: 0.02, tier_c: 0.03},
        ),
        # ... one TierOption per step
    ])

    # Sensitivity-screened targeted grid (typically 10–20 configs)
    configs = builder.targeted_grid(k=12)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from agent_tco.model import StepConfig, WorkflowConfig
from agent_tco.pricing import TierCost


@dataclass
class TierOption:
    """Per-step tier options for config space construction."""

    step_index: int
    tiers: list[TierCost]
    acr_by_tier: dict[TierCost, float]
    hitl_by_tier: dict[TierCost, float]
    avg_output_tokens: int = 500
    label: str = ""

    def step_config(self, tier: TierCost) -> StepConfig:
        """Return a StepConfig for this step at the given tier."""
        if tier not in self.acr_by_tier:
            raise ValueError(
                f"Tier '{tier.tier_id}' not in acr_by_tier for step {self.step_index}"
            )
        return StepConfig(
            step_index=self.step_index,
            model_tier=tier.tier_id,
            acr=self.acr_by_tier[tier],
            hitl_prob=self.hitl_by_tier.get(tier, 0.0),
            inference_cost=(tier.cost_per_1k_tokens * self.avg_output_tokens / 1000),
        )


class ConfigSpaceBuilder:
    """Build and enumerate workflow configuration spaces.

    Args:
        steps: one TierOption per workflow step, in step_index order.
    """

    def __init__(self, steps: list[TierOption]) -> None:
        if not steps:
            raise ValueError("ConfigSpaceBuilder requires at least one step.")
        self._steps = sorted(steps, key=lambda s: s.step_index)

    def enumerate(self) -> list[WorkflowConfig]:
        """Return the full Cartesian product of tier assignments.

        For K steps with T tiers each, returns T^K WorkflowConfigs.
        For large spaces (>10,000 configs), prefer targeted_grid().
        """
        tier_lists = [s.tiers for s in self._steps]
        configs = []
        for i, combo in enumerate(itertools.product(*tier_lists)):
            config_id = "_".join(f"s{s.step_index}{t.tier_id}" for s, t in zip(self._steps, combo))
            steps = [s.step_config(t) for s, t in zip(self._steps, combo)]
            configs.append(WorkflowConfig(config_id=config_id, steps=steps))
        return configs

    def baseline(self, tier_index: int = -1) -> WorkflowConfig:
        """Return the all-same-tier baseline configuration.

        Args:
            tier_index: index into each step's tier list. Default -1 = last tier
                        (typically the cheapest/lowest quality).
        """
        steps = [s.step_config(s.tiers[tier_index]) for s in self._steps]
        tier_id = self._steps[0].tiers[tier_index].tier_id
        return WorkflowConfig(config_id=f"baseline_{tier_id}", steps=steps)

    def targeted_grid(self, k: int | None = None) -> list[WorkflowConfig]:
        """Return a targeted grid of configurations using critical step sensitivity screening.

        Generates:
        - One all-same-tier config per tier (j1, j2, ... for each tier level)
        - For each step × tier combination where the step is a bottleneck candidate:
          upgrade that step to that tier, keep all others at the cheapest tier.

        This produces all-tier anchors plus single-step upgrades at each position
        × each non-cheapest tier.

        Args:
            k: if provided, return only the top-k configs by sensitivity priority.
               If None, return all single-step-upgrade configs.
        """
        from agent_tco.sensitivity import critical_step_sensitivity

        cheapest_tier = {s.step_index: s.tiers[-1] for s in self._steps}
        configs: list[WorkflowConfig] = []

        # All-same-tier anchors
        for ti, tier in enumerate(self._steps[0].tiers):
            steps = [s.step_config(s.tiers[ti]) for s in self._steps]
            label = f"all_{tier.tier_id}"
            configs.append(WorkflowConfig(config_id=label, steps=steps))

        # Baseline for sensitivity screening
        baseline = self.baseline(tier_index=-1)

        # Sensitivity-ranked single-step upgrades
        ranking = critical_step_sensitivity(baseline)
        step_priority = [idx for idx, _ in ranking]

        upgrade_configs = []
        for step_idx in step_priority:
            step_opt = next(s for s in self._steps if s.step_index == step_idx)
            for tier in step_opt.tiers[:-1]:  # all tiers except cheapest
                steps = [
                    s.step_config(tier if s.step_index == step_idx else cheapest_tier[s.step_index])
                    for s in self._steps
                ]
                label = f"s{step_idx}_{tier.tier_id}"
                upgrade_configs.append(WorkflowConfig(config_id=label, steps=steps))

        configs.extend(upgrade_configs)

        if k is not None:
            # Keep all-tier anchors + top-k upgrade configs
            n_anchors = len(self._steps[0].tiers)
            return configs[:n_anchors] + upgrade_configs[:k]

        return configs
