from __future__ import annotations

import pytest

from agent_tco import StepConfig, WorkflowConfig
from agent_tco.config_space import ConfigSpaceBuilder, TierOption
from agent_tco.pricing import TierCost

TIER_A = TierCost("A", 0.50)
TIER_B = TierCost("B", 0.08)
TIER_C = TierCost("C", 0.003)


def _make_option(step_index: int) -> TierOption:
    return TierOption(
        step_index=step_index,
        tiers=[TIER_A, TIER_B, TIER_C],
        acr_by_tier={TIER_A: 0.98, TIER_B: 0.90, TIER_C: 0.80},
        hitl_by_tier={TIER_A: 0.01, TIER_B: 0.02, TIER_C: 0.05},
    )


@pytest.fixture
def builder_3step() -> ConfigSpaceBuilder:
    return ConfigSpaceBuilder(steps=[_make_option(i) for i in range(1, 4)])


class TestConfigSpaceBuilderEnumerate:
    def test_full_enumeration_count(self, builder_3step):
        configs = builder_3step.enumerate()
        assert len(configs) == 3 ** 3  # 27

    def test_all_workflow_configs(self, builder_3step):
        configs = builder_3step.enumerate()
        assert all(isinstance(c, WorkflowConfig) for c in configs)

    def test_unique_config_ids(self, builder_3step):
        configs = builder_3step.enumerate()
        ids = [c.config_id for c in configs]
        assert len(set(ids)) == len(ids)

    def test_each_config_has_correct_step_count(self, builder_3step):
        configs = builder_3step.enumerate()
        for c in configs:
            assert c.n_steps == 3

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError, match="at least one step"):
            ConfigSpaceBuilder(steps=[])


class TestConfigSpaceBuilderBaseline:
    def test_baseline_uses_last_tier(self, builder_3step):
        baseline = builder_3step.baseline()
        for step in baseline.steps:
            assert step.model_tier == "C"

    def test_baseline_first_tier(self, builder_3step):
        baseline = builder_3step.baseline(tier_index=0)
        for step in baseline.steps:
            assert step.model_tier == "A"


class TestConfigSpaceBuilderTargetedGrid:
    def test_targeted_grid_returns_list(self, builder_3step):
        configs = builder_3step.targeted_grid()
        assert isinstance(configs, list)
        assert len(configs) > 0

    def test_targeted_grid_all_same_tier_anchors_present(self, builder_3step):
        configs = builder_3step.targeted_grid()
        ids = [c.config_id for c in configs]
        # Should include all-A, all-B, all-C anchors
        assert any("all_A" in i for i in ids)
        assert any("all_B" in i for i in ids)
        assert any("all_C" in i for i in ids)

    def test_targeted_grid_k_limits_upgrade_configs(self, builder_3step):
        configs_k3 = builder_3step.targeted_grid(k=3)
        configs_k1 = builder_3step.targeted_grid(k=1)
        # k=1 should return fewer upgrade configs than k=3
        n_anchors = 3  # one per tier
        assert len(configs_k1) <= len(configs_k3)

    def test_targeted_grid_unique_ids(self, builder_3step):
        configs = builder_3step.targeted_grid()
        ids = [c.config_id for c in configs]
        assert len(set(ids)) == len(ids)

    def test_targeted_grid_valid_workflow_configs(self, builder_3step):
        configs = builder_3step.targeted_grid()
        for c in configs:
            assert isinstance(c, WorkflowConfig)
            assert c.n_steps == 3

    def test_single_step_builder(self):
        builder = ConfigSpaceBuilder(steps=[_make_option(1)])
        configs = builder.targeted_grid()
        assert len(configs) >= 1

    def test_tier_option_missing_acr_raises(self):
        opt = TierOption(
            step_index=1,
            tiers=[TIER_A, TIER_C],
            acr_by_tier={TIER_A: 0.98},  # TIER_C missing
            hitl_by_tier={TIER_A: 0.01, TIER_C: 0.05},
        )
        builder = ConfigSpaceBuilder(steps=[opt])
        with pytest.raises((KeyError, ValueError)):
            builder.enumerate()
