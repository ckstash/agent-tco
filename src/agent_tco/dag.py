"""Profile-based discrete-event simulation for finite agent execution graphs.

The simulator is a screening tool. It estimates ACR and expected task cost from
declared task/model profiles and execution policies; it does not replace measured
validation of shortlisted assignments in the deployed agent stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
import math
import random
from statistics import fmean, stdev
from typing import Mapping, Sequence

from agent_tco.grid import _wilson_ci
from agent_tco.model import CostParams, Scenario, StepOutcome


class JoinPolicy(Enum):
    """Condition under which incoming activations start a task."""

    ALL = "all"
    ANY = "any"


class OutgoingPolicy(Enum):
    """How matching outgoing edges are activated."""

    FAN_OUT = "fan_out"
    ROUTE = "route"


class SuccessPolicy(Enum):
    """Condition under which declared success nodes complete the workflow."""

    ALL = "all"
    ANY = "any"


class CancellationPolicy(Enum):
    """Billing approximation for a task cancelled by a first-success race."""

    FULL = "full"
    PRORATED = "prorated"
    FREE = "free"


class NodeStatus(Enum):
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TaskProfile:
    """Categorical behavior, cost, and duration for one task/model option."""

    profile_id: str
    q: float
    hitl_prob: float
    partial_prob: float
    invocation_cost: float
    duration_seconds: float = 1.0
    duration_jitter: float = 0.0
    failure_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("q", self.q),
            ("hitl_prob", self.hitl_prob),
            ("partial_prob", self.partial_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]; got {value}")
        if self.q + self.hitl_prob + self.partial_prob > 1.0 + 1e-12:
            raise ValueError("q + hitl_prob + partial_prob must be <= 1")
        if self.invocation_cost < 0.0:
            raise ValueError("invocation_cost must be non-negative")
        if self.duration_seconds < 0.0 or self.duration_jitter < 0.0:
            raise ValueError("duration values must be non-negative")
        if not self.profile_id:
            raise ValueError("profile_id must not be empty")

    @property
    def incomplete_prob(self) -> float:
        return max(0.0, 1.0 - self.q - self.hitl_prob - self.partial_prob)

    def sample_outcome(self, rng: random.Random, forced_failure: bool = False) -> StepOutcome:
        if forced_failure:
            return StepOutcome.INCOMPLETE
        draw = rng.random()
        if draw < self.q:
            return StepOutcome.AUTONOMOUS
        if draw < self.q + self.hitl_prob:
            return StepOutcome.HITL_SUCCESS
        if draw < self.q + self.hitl_prob + self.partial_prob:
            return StepOutcome.HITL_PARTIAL
        return StepOutcome.INCOMPLETE

    def sample_duration(self, rng: random.Random) -> float:
        if self.duration_jitter == 0.0:
            return self.duration_seconds
        low = max(0.0, self.duration_seconds - self.duration_jitter)
        high = self.duration_seconds + self.duration_jitter
        return rng.uniform(low, high)


@dataclass(frozen=True)
class FailureDomain:
    """A run-level common cause that forces member task attempts to fail."""

    domain_id: str
    probability: float
    members: frozenset[str]

    def __post_init__(self) -> None:
        if not self.domain_id:
            raise ValueError("domain_id must not be empty")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("failure-domain probability must be in [0, 1]")
        if not self.members:
            raise ValueError("failure domain requires at least one member")


@dataclass(frozen=True)
class DAGNode:
    """One executable task and its graph-control policies."""

    node_id: str
    join_policy: JoinPolicy = JoinPolicy.ALL
    outgoing_policy: OutgoingPolicy = OutgoingPolicy.FAN_OUT
    max_attempts: int = 1
    max_visits: int = 1
    race_group: str | None = None
    cancellation_policy: CancellationPolicy = CancellationPolicy.FULL

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must not be empty")
        if self.max_attempts < 1 or self.max_visits < 1:
            raise ValueError("max_attempts and max_visits must be >= 1")


@dataclass(frozen=True)
class DAGEdge:
    """A conditional graph transition.

    Route weights may be supplied globally through ``probability`` or by context
    class. Probabilities are used only when the source has ``ROUTE`` policy.
    """

    source: str
    target: str
    on: tuple[StepOutcome, ...] = (StepOutcome.AUTONOMOUS, StepOutcome.HITL_SUCCESS)
    probability: float = 1.0
    context_probabilities: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.on:
            raise ValueError("edge requires at least one triggering outcome")
        if self.probability < 0.0:
            raise ValueError("edge probability must be non-negative")
        if any(value < 0.0 for value in self.context_probabilities.values()):
            raise ValueError("context route probabilities must be non-negative")

    def weight(self, context_class: str | None) -> float:
        if context_class is not None and context_class in self.context_probabilities:
            return self.context_probabilities[context_class]
        return self.probability


@dataclass(frozen=True)
class WorkflowGraph:
    """A finite execution graph with explicit termination guards."""

    graph_id: str
    nodes: Mapping[str, DAGNode]
    edges: tuple[DAGEdge, ...]
    entry_nodes: tuple[str, ...]
    success_nodes: tuple[str, ...]
    success_policy: SuccessPolicy = SuccessPolicy.ALL
    failure_domains: tuple[FailureDomain, ...] = ()
    max_events: int = 10_000
    max_time_seconds: float = 86_400.0

    def __post_init__(self) -> None:
        if not self.graph_id or not self.nodes:
            raise ValueError("graph_id and nodes are required")
        if set(self.nodes) != {node.node_id for node in self.nodes.values()}:
            raise ValueError("node mapping keys must equal DAGNode.node_id values")
        if not self.entry_nodes or not self.success_nodes:
            raise ValueError("entry_nodes and success_nodes are required")
        unknown = (set(self.entry_nodes) | set(self.success_nodes)) - set(self.nodes)
        for edge in self.edges:
            unknown.update({edge.source, edge.target} - set(self.nodes))
        if unknown:
            raise ValueError(f"unknown graph node IDs: {sorted(unknown)}")
        domain_ids = [domain.domain_id for domain in self.failure_domains]
        if len(domain_ids) != len(set(domain_ids)):
            raise ValueError("failure-domain IDs must be unique")
        domain_unknown = set().union(*(domain.members for domain in self.failure_domains)) - set(self.nodes)
        if domain_unknown:
            raise ValueError(f"unknown failure-domain members: {sorted(domain_unknown)}")
        if self.max_events < 1 or self.max_time_seconds <= 0.0:
            raise ValueError("max_events and max_time_seconds must be positive")


@dataclass(frozen=True)
class NodeExecution:
    """One completed or cancelled task attempt in a graph run."""

    node_id: str
    profile_id: str
    visit: int
    attempt: int
    status: NodeStatus
    outcome: StepOutcome | None
    start_time: float
    finish_time: float
    charged_cost: float


@dataclass(frozen=True)
class DAGRun:
    """One simulated complete graph trajectory."""

    config_id: str
    scenario: Scenario
    autonomous_success: bool
    completed: bool
    partial: bool
    assisted: bool
    retry_count: int
    total_cost: float
    latency_seconds: float
    event_count: int
    termination_reason: str
    executions: tuple[NodeExecution, ...]


@dataclass(frozen=True)
class DAGSimulationResult:
    """Monte Carlo aggregate for one graph assignment."""

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
    scenario_probabilities: Mapping[Scenario, float]
    evaluation_mode: str = "profile_dag_monte_carlo"
    search_complete: bool = False
    assumptions: tuple[str, ...] = (
        "declared task profiles represent the target workload",
        "route, retry, join, race, cancellation, and common-cause policies are explicit",
    )


@dataclass
class _ActiveTask:
    node_id: str
    token: int
    visit: int
    attempt: int
    start_time: float
    duration: float
    outcome: StepOutcome
    charged_at_start: float


class DAGSimulator:
    """Execute profile-based graph trajectories using a discrete-event queue."""

    def __init__(self, graph: WorkflowGraph, cost_params: CostParams, seed: int = 42) -> None:
        self.graph = graph
        self.cost_params = cost_params
        self.seed = seed
        self._incoming: dict[str, set[str]] = {node_id: set() for node_id in graph.nodes}
        self._outgoing: dict[str, list[DAGEdge]] = {node_id: [] for node_id in graph.nodes}
        for edge in graph.edges:
            self._incoming[edge.target].add(edge.source)
            self._outgoing[edge.source].append(edge)

    def simulate_run(
        self,
        assignment: Mapping[str, TaskProfile],
        run_index: int = 0,
        context_class: str | None = None,
        config_id: str | None = None,
    ) -> DAGRun:
        required = set(self.graph.nodes)
        if set(assignment) != required:
            raise ValueError(
                f"assignment mismatch: missing={sorted(required-set(assignment))}, "
                f"extra={sorted(set(assignment)-required)}"
            )
        known_domains = {domain.domain_id for domain in self.graph.failure_domains}
        unknown_profile_domains = {
            domain_id
            for profile in assignment.values()
            for domain_id in profile.failure_domains
            if domain_id not in known_domains
        }
        if unknown_profile_domains:
            raise ValueError(
                f"unknown profile failure domains: {sorted(unknown_profile_domains)}"
            )
        rng = random.Random(self.seed + run_index)
        failed_domains = {
            domain.domain_id
            for domain in self.graph.failure_domains
            if rng.random() < domain.probability
        }
        failed_members = set().union(
            *(domain.members for domain in self.graph.failure_domains if domain.domain_id in failed_domains)
        ) if failed_domains else set()

        queue: list[tuple[float, int, str, int]] = []
        active: dict[str, _ActiveTask] = {}
        arrivals: dict[str, set[str]] = {node_id: set() for node_id in self.graph.nodes}
        visits: dict[str, int] = {node_id: 0 for node_id in self.graph.nodes}
        statuses: dict[str, NodeStatus] = {node_id: NodeStatus.NOT_STARTED for node_id in self.graph.nodes}
        success_reached: set[str] = set()
        executions: list[NodeExecution] = []
        sequence = 0
        total_cost = 0.0
        current_time = 0.0
        retry_count = 0
        assisted = False
        partial = False
        completed = False
        termination_reason = "event_queue_exhausted"

        def sample_attempt(node_id: str, visit: int, attempt: int, now: float) -> None:
            nonlocal sequence, total_cost
            node = self.graph.nodes[node_id]
            profile = assignment[node_id]
            sequence += 1
            duration = profile.sample_duration(rng)
            forced_failure = node_id in failed_members or bool(
                set(profile.failure_domains) & failed_domains
            )
            outcome = profile.sample_outcome(rng, forced_failure)
            charged = profile.invocation_cost if node.cancellation_policy == CancellationPolicy.FULL else 0.0
            total_cost += charged
            task = _ActiveTask(node_id, sequence, visit, attempt, now, duration, outcome, charged)
            active[node_id] = task
            statuses[node_id] = NodeStatus.ACTIVE
            heapq.heappush(queue, (now + duration, sequence, node_id, sequence))

        def try_start(node_id: str, predecessor: str | None, now: float) -> None:
            if predecessor is not None:
                arrivals[node_id].add(predecessor)
            node = self.graph.nodes[node_id]
            if node_id in active or visits[node_id] >= node.max_visits:
                return
            ready = predecessor is None
            if predecessor is not None:
                ready = node.join_policy == JoinPolicy.ANY or self._incoming[node_id] <= arrivals[node_id]
            if not ready:
                return
            visits[node_id] += 1
            arrivals[node_id].clear()
            sample_attempt(node_id, visits[node_id], 1, now)

        def cancellation_charge(task: _ActiveTask, now: float) -> float:
            node = self.graph.nodes[task.node_id]
            profile = assignment[task.node_id]
            if node.cancellation_policy == CancellationPolicy.FULL:
                return 0.0
            if node.cancellation_policy == CancellationPolicy.FREE:
                return 0.0
            fraction = 1.0 if task.duration == 0.0 else min(1.0, max(0.0, (now-task.start_time)/task.duration))
            return profile.invocation_cost * fraction

        def cancel_task(node_id: str, now: float) -> None:
            nonlocal total_cost
            task = active.pop(node_id, None)
            if task is None:
                return
            charge = cancellation_charge(task, now)
            total_cost += charge
            statuses[node_id] = NodeStatus.CANCELLED
            executions.append(NodeExecution(
                node_id, assignment[node_id].profile_id, task.visit, task.attempt,
                NodeStatus.CANCELLED, None, task.start_time, now, task.charged_at_start + charge,
            ))

        def choose_edges(node_id: str, outcome: StepOutcome) -> list[DAGEdge]:
            candidates = [edge for edge in self._outgoing[node_id] if outcome in edge.on]
            if not candidates or self.graph.nodes[node_id].outgoing_policy == OutgoingPolicy.FAN_OUT:
                return candidates
            weights = [edge.weight(context_class) for edge in candidates]
            total = sum(weights)
            if total <= 0.0:
                raise ValueError(f"route weights sum to zero at node {node_id!r}")
            draw = rng.random() * total
            cumulative = 0.0
            for edge, weight in zip(candidates, weights):
                cumulative += weight
                if draw <= cumulative:
                    return [edge]
            return [candidates[-1]]

        def success_complete() -> bool:
            declared = set(self.graph.success_nodes)
            if self.graph.success_policy == SuccessPolicy.ANY:
                return bool(success_reached & declared)
            return declared <= success_reached

        for entry in self.graph.entry_nodes:
            try_start(entry, None, 0.0)

        while queue:
            finish_time, _, node_id, token = heapq.heappop(queue)
            task = active.get(node_id)
            if task is None or task.token != token:
                continue
            current_time = finish_time
            if current_time > self.graph.max_time_seconds or len(executions) >= self.graph.max_events:
                termination_reason = "resource_guard"
                break
            active.pop(node_id)
            node = self.graph.nodes[node_id]
            profile = assignment[node_id]
            outcome = task.outcome
            charge = 0.0 if node.cancellation_policy == CancellationPolicy.FULL else profile.invocation_cost
            total_cost += charge
            if outcome in (StepOutcome.HITL_SUCCESS, StepOutcome.HITL_PARTIAL):
                total_cost += self.cost_params.hitl_cost_per_event
                assisted = True

            if outcome == StepOutcome.INCOMPLETE and task.attempt < node.max_attempts:
                executions.append(NodeExecution(
                    node_id, profile.profile_id, task.visit, task.attempt, NodeStatus.FAILED,
                    outcome, task.start_time, current_time, task.charged_at_start + charge,
                ))
                retry_count += 1
                sample_attempt(node_id, task.visit, task.attempt + 1, current_time)
                continue

            status = NodeStatus.PASSED if outcome in (
                StepOutcome.AUTONOMOUS, StepOutcome.HITL_SUCCESS
            ) else NodeStatus.FAILED
            statuses[node_id] = status
            executions.append(NodeExecution(
                node_id, profile.profile_id, task.visit, task.attempt, status, outcome,
                task.start_time, current_time, task.charged_at_start + charge
                + (self.cost_params.hitl_cost_per_event if outcome in (
                    StepOutcome.HITL_SUCCESS, StepOutcome.HITL_PARTIAL
                ) else 0.0),
            ))

            if outcome == StepOutcome.HITL_PARTIAL:
                partial = True
                termination_reason = "partial"
                break

            if status == NodeStatus.PASSED:
                success_reached.add(node_id)
                if node.race_group is not None:
                    for sibling_id, sibling in list(active.items()):
                        if self.graph.nodes[sibling_id].race_group == node.race_group:
                            cancel_task(sibling_id, current_time)
                if success_complete():
                    completed = True
                    termination_reason = "success"
                    break

            for edge in choose_edges(node_id, outcome):
                try_start(edge.target, node_id, current_time)

        for node_id in list(active):
            cancel_task(node_id, current_time)

        if not completed and not partial:
            total_cost += self.cost_params.manual_process_cost
            termination_reason = termination_reason if termination_reason == "resource_guard" else "aborted"

        autonomous_success = completed and not assisted
        if not completed:
            scenario = Scenario.HITL_PARTIAL if partial else Scenario.ABORTED
        elif assisted:
            scenario = Scenario.COMPOUND_RECOVERY if retry_count else Scenario.HITL_SUCCESS
        else:
            scenario = Scenario.RETRY_RESOLVED if retry_count else Scenario.HAPPY_PATH
        identifier = config_id or "|".join(assignment[node_id].profile_id for node_id in self.graph.nodes)
        return DAGRun(
            identifier, scenario, autonomous_success, completed, partial, assisted,
            retry_count, total_cost, current_time, len(executions), termination_reason,
            tuple(executions),
        )

    def evaluate(
        self,
        assignment: Mapping[str, TaskProfile],
        n_runs: int = 1_000,
        context_classes: Sequence[str | None] | None = None,
        config_id: str | None = None,
    ) -> DAGSimulationResult:
        if n_runs < 1:
            raise ValueError("n_runs must be >= 1")
        contexts = context_classes or (None,)
        if not contexts:
            raise ValueError("context_classes must not be empty")
        runs = [
            self.simulate_run(
                assignment, index, contexts[index % len(contexts)], config_id
            )
            for index in range(n_runs)
        ]
        autonomous = sum(run.autonomous_success for run in runs)
        acr = autonomous / n_runs
        acr_low, acr_high = _wilson_ci(autonomous, n_runs)
        costs = [run.total_cost for run in runs]
        expected_tco = fmean(costs)
        if n_runs > 1:
            radius = 1.959963984540054 * stdev(costs) / math.sqrt(n_runs)
        else:
            radius = 0.0
        counts = {scenario: 0 for scenario in Scenario}
        for run in runs:
            counts[run.scenario] += 1
        identifier = runs[0].config_id
        return DAGSimulationResult(
            config_id=identifier,
            assignment=dict(assignment),
            n_runs=n_runs,
            acr=acr,
            expected_tco=expected_tco,
            expected_latency=fmean(run.latency_seconds for run in runs),
            executive_metric=expected_tco/acr if acr else float("inf"),
            acr_ci_lower=acr_low,
            acr_ci_upper=acr_high,
            tco_ci_lower=max(0.0, expected_tco-radius),
            tco_ci_upper=expected_tco+radius,
            scenario_probabilities={key: value/n_runs for key, value in counts.items()},
        )
