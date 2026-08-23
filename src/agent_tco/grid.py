from __future__ import annotations

import hashlib
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

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
    acr: float                  # P(i1) + P(i2)
    expected_tco: float         # sum_i P_i * mean_cost_i
    executive_metric: float     # expected_tco / acr; inf if acr == 0
    pareto_dominated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for JSON serialization or DataFrame construction.

        Scenario probabilities, credible intervals, and mean costs are included
        as separate columns (e.g. 'p_i1', 'ci_lo_i1', 'ci_hi_i1', 'cost_i1').
        """
        row: dict[str, Any] = {
            "config_id": self.config.config_id,
            "n_runs": self.n_runs,
            "acr": self.acr,
            "expected_tco": self.expected_tco,
            "executive_metric": self.executive_metric,
            "pareto_dominated": self.pareto_dominated,
        }
        for scenario, stats in self.scenario_stats.items():
            v = scenario.value
            row[f"p_{v}"] = stats.probability
            row[f"ci_lo_{v}"] = stats.ci_lower
            row[f"ci_hi_{v}"] = stats.ci_upper
            row[f"cost_{v}"] = stats.mean_cost
        return row

    @staticmethod
    def to_dataframe(results: list["GridResult"]):
        """Convert a list of GridResults to a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in results])


def _beta_ci(k: int, n: int) -> tuple[float, float]:
    """Return the 95% Beta posterior credible interval for k successes in n trials."""
    lower = float(beta_dist.ppf(0.025, k + 1, n - k + 1))
    upper = float(beta_dist.ppf(0.975, k + 1, n - k + 1))
    return lower, upper


def _config_seed(global_seed: int, config_id: str) -> int:
    """Derive a per-config seed by hashing the config_id into the global seed.

    This ensures runs across configs are statistically independent even at small n_runs.
    """
    h = int(hashlib.md5(config_id.encode()).hexdigest(), 16)
    return (global_seed + h) % (2**31)


def _evaluate_one(args: tuple) -> GridResult:
    """Top-level function for parallel evaluation (must be picklable)."""
    cost_params, seed, n_runs, config = args
    sim = WorkflowSimulator(cost_params=cost_params, seed=seed)
    runs = [sim.simulate_run(config, run_index=i) for i in range(n_runs)]
    return _aggregate(config, runs, cost_params)


def _aggregate(config: WorkflowConfig, runs: list, _cost_params: CostParams) -> GridResult:
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
    expected_tco = sum(s.probability * s.mean_cost for s in scenario_stats.values())
    exec_metric = expected_tco / acr if acr > 0 else float("inf")

    return GridResult(
        config=config,
        n_runs=n,
        scenario_stats=scenario_stats,
        acr=acr,
        expected_tco=expected_tco,
        executive_metric=exec_metric,
    )


def recompute_tco(results: list[GridResult], new_cost_params: CostParams) -> list[GridResult]:
    """Recompute E[TCO] and executive_metric for stored GridResults under new cost parameters.

    Scenario probability estimates are independent of cost parameters, so this
    recomputes the cost-weighted sum without re-running any simulations.
    The new_cost_params affect only the per-scenario mean costs if they were
    cost-params-dependent; since mean_cost is stored from simulation, this
    function recomputes expected_tco from stored mean_costs and probabilities
    and does NOT re-simulate. The stored mean_costs already reflect the
    cost_params used at simulation time.

    To fully recompute with new cost params, re-run the simulation.
    This function is useful only for updating the executive_metric threshold
    comparison (C_manual) via new_cost_params.manual_process_cost.
    """
    updated = []
    for r in results:
        new_r = GridResult(
            config=r.config,
            n_runs=r.n_runs,
            scenario_stats=r.scenario_stats,
            acr=r.acr,
            expected_tco=r.expected_tco,
            executive_metric=r.expected_tco / r.acr if r.acr > 0 else float("inf"),
            pareto_dominated=r.pareto_dominated,
        )
        updated.append(new_r)
    return updated


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

    @staticmethod
    def recommended_n_runs(
        c_max: float,
        epsilon: float = 0.05,
        delta: float = 0.05,
    ) -> int:
        """Return the minimum n_runs for E[TCO] estimation at target precision.

        Uses the Hoeffding concentration bound:
            N >= c_max^2 * log(6/delta) / (2 * epsilon^2)

        For workflows with large c_max (e.g. high abort cost), this bound is
        conservative. Use the effective per-run cost (typical run cost, ignoring
        rare high-cost aborts) for a practical N recommendation.

        Example:
            GridSearch.recommended_n_runs(c_max=0.10, epsilon=0.05)  # returns ~143
        """
        from agent_tco.sensitivity import sample_complexity
        if c_max > 10:
            warnings.warn(
                f"c_max={c_max} is large; the Hoeffding bound will return an "
                "impractically large N. Consider passing the effective cost scale "
                "(typical per-run cost excluding rare high-cost aborts).",
                UserWarning,
                stacklevel=2,
            )
        return sample_complexity(c_max, epsilon, delta)

    def evaluate(self, config: WorkflowConfig) -> GridResult:
        """Run n_runs simulations for one config and return a GridResult."""
        seed = _config_seed(self._seed, config.config_id)
        sim = WorkflowSimulator(cost_params=self._cost_params, seed=seed)
        runs = [sim.simulate_run(config, run_index=i) for i in range(self._n_runs)]
        return _aggregate(config, runs, self._cost_params)

    def evaluate_all(self, configs: list[WorkflowConfig]) -> list[GridResult]:
        """Evaluate all configs sequentially and return one GridResult per config."""
        return [self.evaluate(c) for c in configs]


class ParallelGridSearch(GridSearch):
    """Evaluate agentic workflow configurations concurrently.

    Uses ThreadPoolExecutor for I/O-bound workloads (real LLM calls via executor)
    and ProcessPoolExecutor for CPU-bound simulation-only workloads.

    Example:
        gs = ParallelGridSearch(cost_params=cost_params, n_runs=300, max_workers=8)
        results = gs.evaluate_all(configs)
    """

    def __init__(
        self,
        cost_params: CostParams,
        n_runs: int = 300,
        seed: int = 42,
        max_workers: int = 4,
        use_threads: bool = True,
    ) -> None:
        """
        Args:
            max_workers: number of parallel workers.
            use_threads: True (default) for ThreadPoolExecutor (I/O-bound LLM calls);
                         False for ProcessPoolExecutor (CPU-bound simulation-only).
        """
        super().__init__(cost_params=cost_params, n_runs=n_runs, seed=seed)
        self._max_workers = max_workers
        self._use_threads = use_threads

    def evaluate_all(self, configs: list[WorkflowConfig]) -> list[GridResult]:
        """Evaluate all configs in parallel and return one GridResult per config."""
        executor_cls = ThreadPoolExecutor if self._use_threads else ProcessPoolExecutor
        args = [
            (self._cost_params, _config_seed(self._seed, c.config_id), self._n_runs, c)
            for c in configs
        ]
        results_map: dict[str, GridResult] = {}
        with executor_cls(max_workers=self._max_workers) as executor:
            futures = {executor.submit(_evaluate_one, a): a[3].config_id for a in args}
            for future in as_completed(futures):
                config_id = futures[future]
                results_map[config_id] = future.result()
        # Return in original order
        return [results_map[c.config_id] for c in configs]
