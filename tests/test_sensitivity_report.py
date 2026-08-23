from __future__ import annotations

import pytest

from agent_tco import WorkflowConfig
from agent_tco.sensitivity import sensitivity_report


@pytest.fixture
def five_step_config(config_j3) -> WorkflowConfig:
    return config_j3


class TestSensitivityReport:
    def test_returns_string(self, five_step_config):
        report = sensitivity_report(five_step_config)
        assert isinstance(report, str)

    def test_contains_all_steps(self, five_step_config):
        report = sensitivity_report(five_step_config)
        for step in five_step_config.steps:
            assert str(step.step_index) in report

    def test_header_present(self, five_step_config):
        report = sensitivity_report(five_step_config)
        assert "Rank" in report
        assert "Sensitivity" in report

    def test_bottleneck_marker_present(self, five_step_config):
        report = sensitivity_report(five_step_config)
        assert "bottleneck" in report

    def test_custom_labels_shown(self, five_step_config):
        labels = {s.step_index: f"MyStep{s.step_index}" for s in five_step_config.steps}
        report = sensitivity_report(five_step_config, step_labels=labels)
        for label in labels.values():
            assert label in report

    def test_step2_is_rank1(self, five_step_config):
        """Step 2 has the lowest ACR in conftest (0.78) so it should be rank 1."""
        report = sensitivity_report(five_step_config)
        lines = [l for l in report.splitlines() if "bottleneck" in l]
        assert len(lines) == 1
        assert "2" in lines[0]

    def test_single_step_workflow(self):
        from agent_tco import StepConfig
        single = WorkflowConfig("single", [StepConfig(1, "C", acr=0.9, hitl_prob=0.05, inference_cost=0.003)])
        report = sensitivity_report(single)
        assert "1" in report
        assert "bottleneck" in report
