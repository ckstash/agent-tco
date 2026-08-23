from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.figure
import pytest

from agent_tco import CostParams, GridSearch, pareto_frontier
from agent_tco.plot import plot_pareto_frontier, plot_sensitivity, plot_scenario_breakdown


@pytest.fixture
def evaluated_results(cost_params, financial_agent_configs):
    gs = GridSearch(cost_params=cost_params, n_runs=50, seed=42)
    results = gs.evaluate_all(financial_agent_configs)
    pareto_frontier(results)
    return results


class TestPlotParetoFrontier:
    def test_returns_figure(self, evaluated_results):
        fig = plot_pareto_frontier(evaluated_results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_manual_cost(self, evaluated_results, cost_params):
        fig = plot_pareto_frontier(evaluated_results, manual_process_cost=cost_params.manual_process_cost)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_paper_style(self, evaluated_results):
        fig = plot_pareto_frontier(evaluated_results, style="paper")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_single_result(self, cost_params, config_j9):
        gs = GridSearch(cost_params=cost_params, n_runs=30, seed=42)
        results = gs.evaluate_all([config_j9])
        pareto_frontier(results)
        fig = plot_pareto_frontier(results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_label_frontier(self, evaluated_results):
        fig = plot_pareto_frontier(evaluated_results, label_frontier=False)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotSensitivity:
    def test_returns_figure(self, config_j3):
        fig = plot_sensitivity(config_j3)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_labels(self, config_j3):
        labels = {s.step_index: f"Step {s.step_index}" for s in config_j3.steps}
        fig = plot_sensitivity(config_j3, step_labels=labels)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_paper_style(self, config_j3):
        fig = plot_sensitivity(config_j3, style="paper")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_single_step(self):
        from agent_tco import StepConfig, WorkflowConfig
        config = WorkflowConfig("s1", [StepConfig(1, "C", acr=0.9, hitl_prob=0.05, inference_cost=0.003)])
        fig = plot_sensitivity(config)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotScenarioBreakdown:
    def test_returns_figure(self, evaluated_results):
        fig = plot_scenario_breakdown(evaluated_results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_paper_style(self, evaluated_results):
        fig = plot_scenario_breakdown(evaluated_results, style="paper")
        assert isinstance(fig, matplotlib.figure.Figure)
