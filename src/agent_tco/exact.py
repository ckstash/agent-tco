"""Exact evaluation for finite sequential workflows with one retry per step."""
from __future__ import annotations

from collections import defaultdict

from agent_tco.grid import GridResult, ScenarioStats
from agent_tco.model import CostParams, Scenario, WorkflowConfig


def evaluate_exact(config: WorkflowConfig, cost_params: CostParams) -> GridResult:
    """Return exact scenario probabilities and expected costs for a chain.

    The evaluator enumerates the finite categorical outcome tree implied by each
    StepConfig. Exact probabilities replace sampling intervals, so each interval
    endpoint equals its probability and ``n_runs`` is zero.
    """
    totals: dict[Scenario, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    def finish(scenario: Scenario, probability: float, cost: float, review_count: int) -> None:
        totals[scenario][0] += probability
        totals[scenario][1] += probability * cost
        totals[scenario][2] += probability * review_count

    def visit(
        index: int,
        probability: float,
        cost: float,
        retried: bool,
        assisted: bool,
        review_count: int,
    ) -> None:
        if index == len(config.steps):
            if assisted:
                scenario = Scenario.COMPOUND_RECOVERY if retried else Scenario.HITL_SUCCESS
            else:
                scenario = Scenario.RETRY_RESOLVED if retried else Scenario.HAPPY_PATH
            finish(scenario, probability, cost, review_count)
            return

        step = config.steps[index]
        q = step.q
        h = step.hitl_prob
        p = step.partial_prob
        u = step.incomplete_prob
        first_cost = cost + step.inference_cost

        if q:
            visit(index + 1, probability * q, first_cost, retried, assisted, review_count)
        if h:
            visit(
                index + 1,
                probability * h,
                first_cost + cost_params.hitl_cost_per_event,
                retried,
                True,
                review_count + 1,
            )
        if p:
            finish(
                Scenario.HITL_PARTIAL,
                probability * p,
                first_cost + cost_params.hitl_cost_per_event,
                review_count + 1,
            )
        if not u:
            return

        retry_cost = first_cost + step.inference_cost
        if q:
            visit(index + 1, probability * u * q, retry_cost, True, assisted, review_count)
        if h:
            visit(
                index + 1,
                probability * u * h,
                retry_cost + cost_params.hitl_cost_per_event,
                True,
                True,
                review_count + 1,
            )
        if p:
            finish(
                Scenario.HITL_PARTIAL,
                probability * u * p,
                retry_cost + cost_params.hitl_cost_per_event,
                review_count + 1,
            )
        if u:
            finish(
                Scenario.ABORTED,
                probability * u * u,
                retry_cost + cost_params.manual_process_cost,
                review_count,
            )

    visit(0, 1.0, 0.0, False, False, 0)
    stats = {}
    for scenario in Scenario:
        probability, weighted_cost, weighted_review_events = totals[scenario]
        mean_cost = weighted_cost / probability if probability else 0.0
        mean_review_events = weighted_review_events / probability if probability else 0.0
        stats[scenario] = ScenarioStats(
            scenario=scenario,
            probability=probability,
            ci_lower=probability,
            ci_upper=probability,
            mean_cost=mean_cost,
            mean_review_events=mean_review_events,
        )
    acr = stats[Scenario.HAPPY_PATH].probability + stats[Scenario.RETRY_RESOLVED].probability
    expected_tco = sum(value[1] for value in totals.values())
    return GridResult(
        config=config,
        n_runs=0,
        scenario_stats=stats,
        acr=acr,
        expected_tco=expected_tco,
        executive_metric=expected_tco / acr if acr else float("inf"),
        acr_ci_lower=acr,
        acr_ci_upper=acr,
        tco_ci_lower=expected_tco,
        tco_ci_upper=expected_tco,
        cost_params=cost_params,
    )