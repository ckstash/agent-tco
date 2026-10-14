from __future__ import annotations

import hashlib
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from agent_tco.model import CostParams, Scenario, WorkflowConfig
from agent_tco.simulator import WorkflowSimulator


@dataclass
class ScenarioStats:
    """Statistics for one scenario class across N simulation runs."""

    scenario: Scenario
    probability: float   # empirical P_i = count_i / N
    ci_lower: float      # 95% Wilson confidence interval lower bound
    ci_upper: float      # 95% Wilson confidence interval upper bound
    mean_cost: float     # empirical mean cost within this scenario (0.0 if no runs)
    mean_review_events: float = 0.0


@dataclass
class GridResult:
    """Aggregated results for one WorkflowConfig after N simulation runs."""

    config: WorkflowConfig
    n_runs: int
    scenario_stats: dict[Scenario, ScenarioStats]
    acr: float                  # P(i1) + P(i2)
    expected_tco: float         # sum_i P_i * mean_cost_i
    executive_metric: float     # expected_tco / acr; inf if acr == 0
    acr_ci_lower: float = 0.0
    acr_ci_upper: float = 1.0
    tco_ci_lower: float = 0.0
    tco_ci_upper: float = float("inf")
    pareto_dominated: bool = False
    cost_params: CostParams | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for JSON serialization or DataFrame construction.

        Scenario probabilities, Wilson confidence intervals, and mean costs are included
        as separate columns (e.g. 'p_i1', 'ci_lo_i1', 'ci_hi_i1', 'cost_i1').
        """
        row: dict[str, Any] = {
            "config_id": self.config.config_id,
            "n_runs": self.n_runs,
            "acr": self.acr,
            "expected_tco": self.expected_tco,
            "executive_metric": self.executive_metric,
            "acr_ci_lower": self.acr_ci_lower,
            "acr_ci_upper": self.acr_ci_upper,
            "tco_ci_lower": self.tco_ci_lower,
            "tco_ci_upper": self.tco_ci_upper,
            "pareto_dominated": self.pareto_dominated,
        }
        if self.cost_params is not None:
            row.update({
                "labor_rate": self.cost_params.labor_rate,
                "review_time_hours": self.cost_params.review_time_hours,
                "manual_process_cost": self.cost_params.manual_process_cost,
            })
        for scenario, stats in self.scenario_stats.items():
            v = scenario.value
            row[f"p_{v}"] = stats.probability
            row[f"ci_lo_{v}"] = stats.ci_lower
            row[f"ci_hi_{v}"] = stats.ci_upper
            row[f"cost_{v}"] = stats.mean_cost
            row[f"review_events_{v}"] = stats.mean_review_events
        return row

    @staticmethod
    def to_dataframe(results: list["GridResult"]):
        """Convert a list of GridResults to a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in results])


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided 95% Wilson confidence interval for a proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    phat = k / n
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


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


def _aggregate(config: WorkflowConfig, runs: list, cost_params: CostParams) -> GridResult:
    """Compute scenario statistics and composite metrics from a list of SimulationRun."""
    n = len(runs)
    scenario_stats: dict[Scenario, ScenarioStats] = {}

    for scenario in Scenario:
        matching = [r for r in runs if r.scenario == scenario]
        k = len(matching)
        probability = k / n
        ci_lower, ci_upper = _wilson_ci(k, n)
        mean_cost = sum(r.total_cost for r in matching) / k if k > 0 else 0.0
        mean_review_events = sum(r.review_count for r in matching) / k if k > 0 else 0.0
        scenario_stats[scenario] = ScenarioStats(
            scenario=scenario,
            probability=probability,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            mean_cost=mean_cost,
            mean_review_events=mean_review_events,
        )

    acr = (
        scenario_stats[Scenario.HAPPY_PATH].probability
        + scenario_stats[Scenario.RETRY_RESOLVED].probability
    )
    expected_tco = sum(s.probability * s.mean_cost for s in scenario_stats.values())
    exec_metric = expected_tco / acr if acr > 0 else float("inf")
    acr_count = sum(
        1 for run in runs if run.scenario in (Scenario.HAPPY_PATH, Scenario.RETRY_RESOLVED)
    )
    acr_ci_lower, acr_ci_upper = _wilson_ci(acr_count, n)
    c_max = (
        2.0 * sum(step.inference_cost for step in config.steps)
        + len(config.steps) * cost_params.hitl_cost_per_event
        + cost_params.manual_process_cost
    )
    radius = c_max * math.sqrt(math.log(40.0) / (2.0 * n))

    return GridResult(
        config=config,
        n_runs=n,
        scenario_stats=scenario_stats,
        acr=acr,
        expected_tco=expected_tco,
        executive_metric=exec_metric,
        acr_ci_lower=acr_ci_lower,
        acr_ci_upper=acr_ci_upper,
        tco_ci_lower=max(0.0, expected_tco - radius),
        tco_ci_upper=expected_tco + radius,
        cost_params=cost_params,
    )


def recompute_tco(results: list[GridResult], new_cost_params: CostParams) -> list[GridResult]:
    """Reprice stored review and fallback costs without re-running behavior.

    Inference costs remain fixed because they belong to each ``StepConfig``.
    Each scenario stores its mean review-event count, and aborted scenarios
    contain exactly one manual-fallback charge. These statistics and the original
    ``cost_params`` are sufficient to adjust each scenario mean.

    Legacy results without original cost parameters cannot be repriced safely.
    Exact results retain point TCO intervals; Monte Carlo TCO intervals are
    reset because their run-level cost samples are not stored.
    """
    updated = []
    for r in results:
        if r.cost_params is None:
            raise ValueError(
                f"Cannot reprice '{r.config.config_id}': original cost_params are unavailable"
            )
        old_review = r.cost_params.hitl_cost_per_event
        new_review = new_cost_params.hitl_cost_per_event
        old_manual = r.cost_params.manual_process_cost
        new_manual = new_cost_params.manual_process_cost
        repriced_stats: dict[Scenario, ScenarioStats] = {}
        for scenario, stats in r.scenario_stats.items():
            mean_cost = stats.mean_cost
            if stats.probability == 0.0:
                mean_cost = 0.0
            else:
                mean_cost += stats.mean_review_events * (new_review - old_review)
            if stats.probability > 0.0 and scenario == Scenario.ABORTED:
                mean_cost += new_manual - old_manual
            repriced_stats[scenario] = ScenarioStats(
                scenario=scenario,
                probability=stats.probability,
                ci_lower=stats.ci_lower,
                ci_upper=stats.ci_upper,
                mean_cost=mean_cost,
                mean_review_events=stats.mean_review_events,
            )
        expected_tco = sum(
            stats.probability * stats.mean_cost
            for stats in repriced_stats.values()
        )
        exact = r.n_runs == 0
        new_r = GridResult(
            config=r.config,
            n_runs=r.n_runs,
            scenario_stats=repriced_stats,
            acr=r.acr,
            expected_tco=expected_tco,
            executive_metric=expected_tco / r.acr if r.acr > 0 else float("inf"),
            acr_ci_lower=r.acr_ci_lower,
            acr_ci_upper=r.acr_ci_upper,
            tco_ci_lower=expected_tco if exact else 0.0,
            tco_ci_upper=expected_tco if exact else float("inf"),
            pareto_dominated=False,
            cost_params=new_cost_params,
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

        Uses the direct Hoeffding concentration bound. ``c_max`` must bound every
        run cost; a typical or truncated cost does not preserve the guarantee.

        Example:
            GridSearch.recommended_n_runs(c_max=0.10, epsilon=0.05)  # returns 8
        """
        from agent_tco.sensitivity import sample_complexity
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
