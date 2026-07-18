from __future__ import annotations

import random
from typing import Callable

from agent_tco.model import (
    CostParams,
    Scenario,
    SimulationRun,
    StepConfig,
    WorkflowConfig,
)


class WorkflowSimulator:
    """Simulate one run of a workflow under a given configuration.

    Implements the single-retry model: if a step does not complete on the first
    attempt, one retry is made. If the retry also fails, the run is aborted
    (scenario i6) and the opportunity cost is applied.

    Pass an optional executor callable to drive steps with real LLM calls instead
    of Bernoulli sampling. The executor must match StepExecutor protocol:
    (step: StepConfig, context: dict) -> tuple[bool, bool] (completed, hitl_triggered).
    """

    def __init__(
        self,
        cost_params: CostParams,
        seed: int = 42,
        executor: Callable | None = None,
    ) -> None:
        """Initialize the simulator with cost parameters and optional LLM executor."""
        self._cost_params = cost_params
        self._seed = seed
        self._executor = executor

    def simulate_run(
        self,
        config: WorkflowConfig,
        run_index: int = 0,
    ) -> SimulationRun:
        """Simulate one workflow run and return the classified outcome."""
        rng = random.Random(self._seed + run_index)
        cost = 0.0
        retry_count = 0
        hitl_triggered = False
        terminal = "success"
        context: dict = {}

        for step in config.steps:
            cost += step.inference_cost  # first-attempt inference cost

            if self._executor is not None:
                completed, hitl = self._executor(step, context)
            else:
                hitl = rng.random() < step.hitl_prob
                completed = rng.random() < step.acr

            if hitl:
                hitl_triggered = True
                cost += self._cost_params.hitl_cost_per_event

            if not completed:
                # Single retry
                retry_count += 1
                cost += step.inference_cost  # retry inference cost
                if self._executor is not None:
                    completed, _ = self._executor(step, context)
                else:
                    completed = rng.random() < step.acr

                if not completed:
                    cost += self._cost_params.manual_process_cost
                    terminal = "aborted"
                    break

        scenario = self.classify_scenario(retry_count, hitl_triggered, terminal)
        return SimulationRun(
            scenario=scenario,
            total_cost=cost,
            retry_count=retry_count,
            hitl_triggered=hitl_triggered,
        )

    def classify_scenario(
        self,
        retry_count: int,
        hitl_triggered: bool,
        terminal_state: str,
    ) -> Scenario:
        """Classify a run outcome into one of the six scenario classes."""
        if terminal_state == "aborted":
            return Scenario.ABORTED
        if terminal_state == "partial":
            return Scenario.HITL_PARTIAL  # h=1 implied when partial
        # terminal_state == "success"
        if not hitl_triggered and retry_count == 0:
            return Scenario.HAPPY_PATH
        if not hitl_triggered and retry_count >= 1:
            return Scenario.RETRY_RESOLVED
        if hitl_triggered and retry_count == 0:
            return Scenario.HITL_SUCCESS
        # hitl_triggered and retry_count >= 1
        return Scenario.COMPOUND_RECOVERY
