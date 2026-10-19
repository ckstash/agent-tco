# agent-tco

`agent-tco` compares model assignments for agentic workflows on two objectives: Autonomous Completion Rate (ACR) and Expected Total Cost of Ownership (E[TCO]) per incoming task. It supports exact sequential and structured-block evaluation, approximate profile-based execution-graph simulation, and measured validation of shortlisted configurations.

### Cost terminology

The package name and public API use **TCO** as an umbrella term for expected
trajectory cost. The computed boundary is determined by the supplied model and
cost inputs; the library does not automatically add every acquisition,
integration, infrastructure, governance, maintenance, or retirement cost that
may belong to a full lifecycle TCO study.

The [companion paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7621679)
uses the narrower term **expected operating cost (EOC)** for inference, retry,
human-review, and terminal manual-replacement cost per incoming task. With the
paper's supplied profiles, the library value exposed as
`expected_tco` is exactly this EOC quantity. The names `agent-tco`, `agent_tco`,
`expected_tco`, and `recompute_tco()` are retained for package and API
compatibility.

## Outcome model

Each `StepConfig` defines mutually exclusive per-attempt probabilities,
conditional on execution reaching that touchpoint with its required predecessors
in a passing state.

- `q` is autonomous completion on one attempt.
- `hitl_prob` is human-assisted step completion, which advances to downstream work while removing the trajectory from ACR.
- `partial_prob` is human-assisted partial completion.
- The residual `1 - acr - hitl_prob - partial_prob` is retryable non-completion.

The three supplied probabilities must sum to at most one. A non-completion receives one retry. Two consecutive non-completions invoke the configured manual fallback.

### How workflow ACR and E[TCO] compose

For a required series chain with autonomous-passage probabilities `a_k`, workflow
ACR is `prod(a_k)` under the evaluator's conditional-independence assumption.
The structured evaluator recursively applies the policy of each block instead of
blindly multiplying arbitrary subgraphs: eager AND has the series reliability
formula, independent OR has `1 - prod(1 - a_k)`, ordered fallback invokes later
branches only after earlier failures, and bounded retry repeats its child.

Costs add along each executed trajectory and E[TCO] averages those trajectory
costs. Consequently, a stop-on-failure chain with invocation costs `c_k` has
`c_1 + a_1*c_2 + a_1*a_2*c_3 + ...` before review and manual-fallback terms; it
does not normally pay for unreached nodes. Eager parallel blocks do pay for every
invoked branch. This reach-weighted accounting is why topology affects cost as
well as reliability.

## Features

- Monte Carlo simulation with Wilson confidence intervals for scenario probabilities and ACR
- Exact finite-state evaluation for categorical sequential chains
- Full Cartesian enumeration and sensitivity-screened configuration sets
- Pareto-frontier analysis on ACR and E[TCO]
- Hoeffding intervals for expected cost using a true per-run cost bound
- Categorical `StepOutcome` executor hook for measured model calls
- JSON persistence and review/fallback repricing without re-running behavior
- Exact compositional evaluation for nested series, eager AND, ordered-fallback OR, eager OR, and bounded-retry blocks
- Exhaustive and deterministic budgeted guided search for structured-topology assignments
- Discrete-event Monte Carlo for finite execution graphs with shared descendants, joins, routes, bounded cycles, first-success races, cancellation billing, and common-cause failures
- Budgeted DAG assignment screening with uncertainty intervals
- Balanced synchronous or bounded-concurrency asynchronous experiments through a user-supplied real-agent executor
- Per-run measured evidence, measured Pareto extraction, JSON export, and frontier plotting
- Reproducible stylized computational tests of the five conditional hypotheses

## Installation

```bash
pip install agent-tco
```

Python 3.12 or later is required.

## Minimal example

```python
from agent_tco import (
    CostParams,
    StepConfig,
    WorkflowConfig,
    evaluate_exact,
    pareto_frontier,
)

costs = CostParams(
    labor_rate=60.0,
    review_time_hours=0.25,
    manual_process_cost=4.50,
)

lightweight = WorkflowConfig("CC", [
    StepConfig(1, "C", q=0.78, hitl_prob=0.08,
               partial_prob=0.02, inference_cost=0.003),
    StepConfig(2, "C", q=0.74, hitl_prob=0.10,
               partial_prob=0.03, inference_cost=0.003),
])

mixed = WorkflowConfig("BA", [
    StepConfig(1, "B", q=0.88, hitl_prob=0.04,
               partial_prob=0.01, inference_cost=0.08),
    StepConfig(2, "A", q=0.94, hitl_prob=0.02,
               partial_prob=0.005, inference_cost=0.50),
])

results = [evaluate_exact(config, costs) for config in (lightweight, mixed)]
frontier = pareto_frontier(results)
for result in frontier:
    print(result.config.config_id, result.acr, result.expected_tco)
```

`E[TCO]/ACR` is available as `executive_metric`. It is a descriptive cost-effectiveness ratio. Compare E[TCO] directly with manual task cost when both alternatives deliver comparable task value.

## Repricing stored results

`save_results()` stores the original review and manual-fallback cost parameters.
After loading results, `recompute_tco()` can change those two cost components
without repeating workflow evaluation:

```python
from agent_tco import CostParams, load_results, pareto_frontier, recompute_tco

stored = load_results("results.json")
updated_costs = CostParams(
    labor_rate=75.0,
    review_time_hours=0.20,
    manual_process_cost=12.0,
)
repriced = recompute_tco(stored, updated_costs)
frontier = pareto_frontier(repriced)
```

Inference prices are part of each `StepConfig` and are not changed by
`recompute_tco()`. Re-evaluate configurations after changing model behavior,
transition probabilities, retry policy, topology, or inference cost.

## Screening before measured validation

Profiles should be calibrated from historical traces, controlled evaluations, or
pilot runs. Use exact composition or DAG simulation to reduce the assignment
space, then run the shortlist against representative inputs through the actual
agent stack. `ProductionExperimentRunner` and
`AsyncProductionExperimentRunner` randomize a balanced configuration/case
schedule and aggregate observed autonomy, total cost, and latency. The supplied
executor remains responsible for authentication, rate limiting, tracing,
scoring, retries performed by the real agent, and provider billing telemetry.
The measured Pareto frontier—not the simulated frontier—is the appropriate
evidence for a deployment decision.

`run_stylized_hypothesis_tests()` exercises model-implied comparative statics for
convex capability prices, finite reliability ceilings, dependent redundancy,
labor rates, and importance/review alignment. Its synthetic results are
implementation checks and illustrations, not external empirical validation.

## Reproducing the companion paper

The exact synthetic profiles used by *Beyond Point Solutions: Cost-Quality
Optimization of Multi-Model Agentic Workflows* are included in
`agent_tco.paper`. In this section, values printed from `expected_tco` are the
paper's EOC values under its stated operating-cost boundary.

From the repository root, install the package environment and run:

```bash
uv sync
uv run python - <<'PY'
from agent_tco import (
    evaluate_paper_chain,
    evaluate_paper_structured_exhaustive,
    evaluate_paper_structured_guided,
    pareto_frontier,
)

chain_results = evaluate_paper_chain()
chain_frontier = pareto_frontier(chain_results)
chain_best = min(chain_frontier, key=lambda result: result.executive_metric)
print(
    "chain", len(chain_results), len(chain_frontier),
    chain_best.config.config_id,
    f"ACR={chain_best.acr:.3f}",
    f"EOC=${chain_best.expected_tco:.2f}",
    f"EOC/ACR=${chain_best.executive_metric:.2f}",
)

structured = evaluate_paper_structured_exhaustive()
structured_best = min(
    structured.frontier,
    key=lambda result: result.executive_metric,
)
guided = evaluate_paper_structured_guided()
guided_recovers_frontier = {
    result.config_id for result in guided.frontier
} == {
    result.config_id for result in structured.frontier
}
print(
    "structured", len(structured.results), len(structured.frontier),
    structured_best.config_id,
    f"ACR={structured_best.acr:.3f}",
    f"EOC=${structured_best.expected_tco:.2f}",
    f"EOC/ACR=${structured_best.executive_metric:.2f}",
)
print("guided", guided.evaluated, len(guided.frontier), guided_recovers_frontier)
PY
```

Expected output:

```text
chain 243 7 BABBB ACR=0.893 EOC=$2.76 EOC/ACR=$3.09
structured 6561 8 BAAAABAA ACR=0.725 EOC=$4.85 EOC/ACR=$6.70
guided 500 8 True
```

The first line reproduces the five-step chain's complete seven-point frontier
and its minimum-ratio assignment. The second line reproduces the structured
workflow's complete eight-point frontier and minimum-ratio assignment. The
third line confirms retrospectively that the 500-evaluation guided search
contains all eight frontier assignments for this constructed instance.

The shorter Python calls below are useful when inspecting each result set
interactively. The five-step example and its full frontier can be reproduced
without copying constants from the manuscript:

```python
from agent_tco import evaluate_paper_chain, pareto_frontier

chain_results = evaluate_paper_chain()
chain_frontier = pareto_frontier(chain_results)
print(len(chain_results), [r.config.config_id for r in chain_frontier])
```

The eight-touchpoint block-tree example exposes both reported searches:

```python
from agent_tco import (
    evaluate_paper_structured_exhaustive,
    evaluate_paper_structured_guided,
)

exact = evaluate_paper_structured_exhaustive()  # all 6,561 assignments
guided = evaluate_paper_structured_guided()     # deterministic budget of 500
print(len(exact.frontier), guided.evaluated)
```

These profiles are synthetic evaluator checks, not calibrated production
estimates. Both reported frontiers use deterministic exact evaluation; Monte
Carlo is not required to reproduce them.

## Scope

The sequential API executes chains with one retry after a non-completion. The
structured API evaluates nested series, eager AND, ordered-fallback OR, eager OR,
and bounded-retry blocks exactly under conditionally independent task outcomes.
The DAG API approximately evaluates finite graphs using declared joins, routes,
visit/attempt limits, duration profiles, cancellation policy, and common-cause
domains. Every cycle must terminate through finite visit/event/time guards;
unbounded hidden-state behavior is not modeled. Guided and DAG searches are
heuristic and report a frontier only over evaluated assignments. Only exhaustive
evaluation certifies its declared finite candidate space under evaluator
assumptions.

## License

MIT License