"""Measured production-like experiments for simulator-shortlisted assignments."""
from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import json
import random
from statistics import fmean, stdev
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from agent_tco.dag import TaskProfile
from agent_tco.grid import _wilson_ci


@dataclass(frozen=True)
class MeasuredRun:
    """Observed outcome and cost from one real agent execution."""

    autonomous_success: bool
    completed: bool
    total_cost: float
    latency_seconds: float
    assisted: bool = False
    retry_count: int = 0
    partial: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_cost < 0.0 or self.latency_seconds < 0.0 or self.retry_count < 0:
            raise ValueError("measured cost, latency, and retry count must be non-negative")
        if self.autonomous_success and (not self.completed or self.assisted):
            raise ValueError("autonomous success requires completed=True and assisted=False")


class ProductionExecutor(Protocol):
    """Adapter that runs the real agent stack for one assignment and input case."""

    def __call__(
        self,
        assignment: Mapping[str, TaskProfile],
        input_case: Any,
        run_id: str,
    ) -> MeasuredRun: ...


class AsyncProductionExecutor(Protocol):
    """Asynchronous adapter for a real agent stack or remote orchestration API."""

    async def __call__(
        self,
        assignment: Mapping[str, TaskProfile],
        input_case: Any,
        run_id: str,
    ) -> MeasuredRun: ...


@dataclass(frozen=True)
class ExperimentRecord:
    run_id: str
    config_id: str
    case_index: int
    repeat: int
    measurement: MeasuredRun


@dataclass(frozen=True)
class MeasuredConfigResult:
    config_id: str
    assignment: Mapping[str, TaskProfile]
    n_runs: int
    acr: float
    expected_tco: float
    expected_latency: float
    executive_metric: float
    acr_ci_lower: float
    acr_ci_upper: float
    tco_ci_lower: float
    tco_ci_upper: float


@dataclass(frozen=True)
class ExperimentReport:
    records: tuple[ExperimentRecord, ...]
    results: tuple[MeasuredConfigResult, ...]
    frontier: tuple[MeasuredConfigResult, ...]
    seed: int

    def save_json(self, path: str | Path) -> None:
        data = {
            "seed": self.seed,
            "records": [
                {
                    "run_id": record.run_id,
                    "config_id": record.config_id,
                    "case_index": record.case_index,
                    "repeat": record.repeat,
                    "measurement": {
                        "autonomous_success": record.measurement.autonomous_success,
                        "completed": record.measurement.completed,
                        "total_cost": record.measurement.total_cost,
                        "latency_seconds": record.measurement.latency_seconds,
                        "assisted": record.measurement.assisted,
                        "retry_count": record.measurement.retry_count,
                        "partial": record.measurement.partial,
                        "metadata": dict(record.measurement.metadata),
                    },
                }
                for record in self.records
            ],
            "results": [result_to_dict(result) for result in self.results],
            "frontier_ids": [result.config_id for result in self.frontier],
        }
        Path(path).write_text(json.dumps(data, indent=2)+"\n")


class ProductionExperimentRunner:
    """Run shortlisted assignments against real representative agent inputs.

    The runner randomizes a balanced assignment/case/repeat schedule. The supplied
    executor owns production concerns such as authentication, rate limiting,
    tracing, scoring, retries inside the real agent, and provider billing capture.
    """

    def __init__(self, executor: ProductionExecutor, seed: int = 42) -> None:
        self.executor = executor
        self.seed = seed

    def run(
        self,
        assignments: Mapping[str, Mapping[str, TaskProfile]],
        input_cases: Sequence[Any],
        repeats: int = 1,
    ) -> ExperimentReport:
        if not assignments or not input_cases or repeats < 1:
            raise ValueError("assignments, input_cases, and positive repeats are required")
        schedule = [
            (config_id, case_index, repeat)
            for repeat in range(repeats)
            for case_index in range(len(input_cases))
            for config_id in assignments
        ]
        random.Random(self.seed).shuffle(schedule)
        records = []
        for index, (config_id, case_index, repeat) in enumerate(schedule):
            run_id = f"run-{index:06d}-{config_id}-c{case_index}-r{repeat}"
            measurement = self.executor(assignments[config_id], input_cases[case_index], run_id)
            records.append(ExperimentRecord(run_id, config_id, case_index, repeat, measurement))
        results = tuple(
            aggregate_measured(
                config_id,
                assignments[config_id],
                [record.measurement for record in records if record.config_id == config_id],
            )
            for config_id in assignments
        )
        return ExperimentReport(tuple(records), results, tuple(measured_pareto_frontier(results)), self.seed)


class AsyncProductionExperimentRunner:
    """Run a balanced measured experiment with bounded asynchronous concurrency."""

    def __init__(
        self,
        executor: AsyncProductionExecutor,
        seed: int = 42,
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.executor = executor
        self.seed = seed
        self.max_concurrency = max_concurrency

    async def run(
        self,
        assignments: Mapping[str, Mapping[str, TaskProfile]],
        input_cases: Sequence[Any],
        repeats: int = 1,
    ) -> ExperimentReport:
        if not assignments or not input_cases or repeats < 1:
            raise ValueError("assignments, input_cases, and positive repeats are required")
        schedule = [
            (config_id, case_index, repeat)
            for repeat in range(repeats)
            for case_index in range(len(input_cases))
            for config_id in assignments
        ]
        random.Random(self.seed).shuffle(schedule)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def execute(index: int, config_id: str, case_index: int, repeat: int):
            run_id = f"run-{index:06d}-{config_id}-c{case_index}-r{repeat}"
            async with semaphore:
                measurement = await self.executor(
                    assignments[config_id], input_cases[case_index], run_id
                )
            return ExperimentRecord(run_id, config_id, case_index, repeat, measurement)

        records = tuple(await asyncio.gather(*(
            execute(index, config_id, case_index, repeat)
            for index, (config_id, case_index, repeat) in enumerate(schedule)
        )))
        results = tuple(
            aggregate_measured(
                config_id,
                assignments[config_id],
                [record.measurement for record in records if record.config_id == config_id],
            )
            for config_id in assignments
        )
        return ExperimentReport(records, results, tuple(measured_pareto_frontier(results)), self.seed)


def aggregate_measured(
    config_id: str,
    assignment: Mapping[str, TaskProfile],
    runs: Sequence[MeasuredRun],
) -> MeasuredConfigResult:
    if not runs:
        raise ValueError("at least one measured run is required")
    n = len(runs)
    autonomous = sum(run.autonomous_success for run in runs)
    acr = autonomous/n
    acr_low, acr_high = _wilson_ci(autonomous, n)
    costs = [run.total_cost for run in runs]
    expected_tco = fmean(costs)
    radius = 1.959963984540054*stdev(costs)/math.sqrt(n) if n > 1 else 0.0
    return MeasuredConfigResult(
        config_id, dict(assignment), n, acr, expected_tco,
        fmean(run.latency_seconds for run in runs),
        expected_tco/acr if acr else float("inf"),
        acr_low, acr_high, max(0.0, expected_tco-radius), expected_tco+radius,
    )


def measured_pareto_frontier(
    results: Sequence[MeasuredConfigResult],
) -> list[MeasuredConfigResult]:
    ordered = sorted(results, key=lambda result: (-result.acr, result.expected_tco, result.config_id))
    frontier = []
    best_cost = float("inf")
    for result in ordered:
        if result.expected_tco < best_cost:
            frontier.append(result)
            best_cost = result.expected_tco
    return sorted(frontier, key=lambda result: (result.expected_tco, result.acr, result.config_id))


def result_to_dict(result: MeasuredConfigResult) -> dict[str, Any]:
    return {
        "config_id": result.config_id,
        "n_runs": result.n_runs,
        "acr": result.acr,
        "expected_tco": result.expected_tco,
        "expected_latency": result.expected_latency,
        "executive_metric": result.executive_metric,
        "acr_ci_lower": result.acr_ci_lower,
        "acr_ci_upper": result.acr_ci_upper,
        "tco_ci_lower": result.tco_ci_lower,
        "tco_ci_upper": result.tco_ci_upper,
    }
