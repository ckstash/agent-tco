from __future__ import annotations

import random
from typing import Callable

from agent_tco.model import (
    CostParams,
    Scenario,
    SimulationRun,
    StepConfig,
    StepOutcome,
    WorkflowConfig,
)


class WorkflowSimulator:
    """Simulate one run of a workflow under a given configuration.

    Implements the single-retry model: if a step does not complete on the first
    attempt, one retry is made. If the retry also fails, the run is aborted
    (scenario i6) and the opportunity cost is applied.

    Each attempt has one categorical outcome. Autonomous and human-assisted
    completion advance to the next step; assisted completion also records review
    cost. Partial completion terminates the run. A non-completion receives one
    retry and then aborts.

    An optional executor can return a StepOutcome. Legacy ``(completed, hitl)``
    tuples remain supported.
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
        review_count = 0
        terminal = "success"
        context: dict = {}

        for step in config.steps:
            cost += step.inference_cost  # first-attempt inference cost

            outcome = self._attempt(step, context, rng)

            if outcome == StepOutcome.HITL_SUCCESS:
                hitl_triggered = True
                review_count += 1
                cost += self._cost_params.hitl_cost_per_event
                continue
            if outcome == StepOutcome.HITL_PARTIAL:
                hitl_triggered = True
                review_count += 1
                cost += self._cost_params.hitl_cost_per_event
                terminal = "partial"
                break

            if outcome == StepOutcome.INCOMPLETE:
                # Single retry
                retry_count += 1
                cost += step.inference_cost  # retry inference cost
                outcome = self._attempt(step, context, rng)
                if outcome == StepOutcome.HITL_SUCCESS:
                    hitl_triggered = True
                    review_count += 1
                    cost += self._cost_params.hitl_cost_per_event
                    continue
                if outcome == StepOutcome.HITL_PARTIAL:
                    hitl_triggered = True
                    review_count += 1
                    cost += self._cost_params.hitl_cost_per_event
                    terminal = "partial"
                    break
                if outcome == StepOutcome.INCOMPLETE:
                    cost += self._cost_params.manual_process_cost
                    terminal = "aborted"
                    break

        scenario = self.classify_scenario(retry_count, hitl_triggered, terminal)
        return SimulationRun(
            scenario=scenario,
            total_cost=cost,
            retry_count=retry_count,
            hitl_triggered=hitl_triggered,
            review_count=review_count,
        )

    def _attempt(self, step: StepConfig, context: dict, rng: random.Random) -> StepOutcome:
        """Execute or sample one attempt and return its categorical outcome."""
        if self._executor is None:
            return step.outcome_from_uniform(rng.random())
        result = self._executor(step, context)
        if isinstance(result, StepOutcome):
            return result
        completed, hitl = result
        if completed and hitl:
            return StepOutcome.HITL_SUCCESS
        if completed:
            return StepOutcome.AUTONOMOUS
        return StepOutcome.INCOMPLETE

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
