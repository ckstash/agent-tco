from __future__ import annotations

from dataclasses import dataclass, field

from scipy.stats import beta as beta_dist

from agent_tco.model import CostParams, Scenario, WorkflowConfig
from agent_tco.simulator import WorkflowSimulator


@dataclass
class ScenarioStats:
    """Statistics for one scenario class across N simulation runs."""

    scenario: Scenario
    probability: float   # empirical P_i = count_i / N
    ci_lower: float      # 95% Beta credible interval lower bound
    ci_upper: float      # 95% Beta credible interval upper bound
    mean_cost: float     # empirical mean cost within this scenario (0.0 if no runs)


@dataclass
class GridResult:
    """Aggregated results for one WorkflowConfig after N simulation runs."""

    config: WorkflowConfig
    n_runs: int
    scenario_stats: dict[Scenario, ScenarioStats]
    acr: float            # P(i1) + P(i2)
    expected_tco: float   # sum_i P_i * mean_cost_i
    executive_metric: float  # expected_tco / acr; inf if acr == 0
    pareto_dominated: bool = False


def _beta_ci(k: int, n: int) -> tuple[float, float]:
    """Return the 95% Beta posterior credible interval for k successes in n trials."""
    lower = float(beta_dist.ppf(0.025, k + 1, n - k + 1))
    upper = float(beta_dist.ppf(0.975, k + 1, n - k + 1))
    return lower, upper


class GridSearch:
    """Evaluate agentic workflow configurations over N simulation runs each."""

    def __init__(
        self,
        cost_params: CostParams,
        n_runs: int = 300,
        seed: int = 42,
    ) -> None:
        """Initialize the grid search with cost parameters and simulation budget."""
        self._cost_params = cost_params
        self._n_runs = n_runs
        self._seed = seed

    def evaluate(self, config: WorkflowConfig) -> GridResult:
        """Run n_runs simulations for one config and return a GridResult."""
        sim = WorkflowSimulator(cost_params=self._cost_params, seed=self._seed)
        runs = [sim.simulate_run(config, run_index=i) for i in range(self._n_runs)]
        return self._aggregate(config, runs)

    def evaluate_all(self, configs: list[WorkflowConfig]) -> list[GridResult]:
        """Evaluate all configs sequentially and return one GridResult per config."""
        return [self.evaluate(c) for c in configs]

    def _aggregate(self, config: WorkflowConfig, runs: list) -> GridResult:
        """Compute scenario statistics and composite metrics from a list of SimulationRun."""
        n = len(runs)
        scenario_stats: dict[Scenario, ScenarioStats] = {}

        for scenario in Scenario:
            matching = [r for r in runs if r.scenario == scenario]
            k = len(matching)
            probability = k / n
            ci_lower, ci_upper = _beta_ci(k, n)
            mean_cost = sum(r.total_cost for r in matching) / k if k > 0 else 0.0
            scenario_stats[scenario] = ScenarioStats(
                scenario=scenario,
                probability=probability,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                mean_cost=mean_cost,
            )

        acr = (
            scenario_stats[Scenario.HAPPY_PATH].probability
            + scenario_stats[Scenario.RETRY_RESOLVED].probability
        )
        expected_tco = sum(
            s.probability * s.mean_cost for s in scenario_stats.values()
        )
        exec_metric = expected_tco / acr if acr > 0 else float("inf")

        return GridResult(
            config=config,
            n_runs=n,
            scenario_stats=scenario_stats,
            acr=acr,
            expected_tco=expected_tco,
            executive_metric=exec_metric,
        )
