# agent-tco

**agent-tco** is a Python library for Pareto-optimal configuration of agentic workflows.

---

## Features

- **Workflow simulator** with six-class scenario taxonomy (i1–i6): happy path, retry-resolved, HITL success, HITL partial, compound recovery, aborted.
- **Grid search** — evaluates all candidate configurations with Beta credible intervals on scenario probabilities.
- **Pareto frontier** — identifies non-dominated configurations on the ACR vs. E[TCO] plane.
- **Critical step sensitivity analysis** — computes ∂ACR/∂q_k to prioritize which step to upgrade first.
- **Tier pricing utilities** — built-in `TIER_A`, `TIER_B`, `TIER_C` constants covering frontier, mid-range, and lightweight models. Pass your own `TierCost(tier_id, cost_per_1k_tokens)` to use current pricing.
- **`StepExecutor` protocol** — plug in real LLM calls or a `Mock` for testing, no SDK dependency.

---

## Installation

```bash
pip install agent-tco
```

Built for Python 3.12 or above.

---

## Quick Start

### 1. Define configurations and run the grid search

```python
from agent_tco import (
    CostParams, GridSearch,
    step_config_from_tier, WorkflowConfig,
    TIER_A, TIER_B, TIER_C,
)

cost_params = CostParams(
    labor_rate=60.0,
    review_time_hours=0.25,
    manual_process_cost=50.0,
)

# j3: all Tier C
j3 = WorkflowConfig("j3", [
    step_config_from_tier(k, TIER_C, acr=0.78 if k == 2 else 0.92, hitl_prob=0.20 if k == 4 else 0.05)
    for k in range(1, 6)
])

# j9: Tier B at step 2 (the bottleneck), Tier C elsewhere
j9 = WorkflowConfig("j9", [
    step_config_from_tier(k, TIER_B if k == 2 else TIER_C,
                          acr=0.88 if k == 2 else 0.92,
                          hitl_prob=0.08 if k == 2 else (0.20 if k == 4 else 0.05))
    for k in range(1, 6)
])

gs = GridSearch(cost_params=cost_params, n_runs=300, seed=42)
results = gs.evaluate_all([j3, j9])
for r in results:
    print(f"{r.config.config_id}: ACR={r.acr:.2f}  E[TCO]=${r.expected_tco:.2f}  M=${r.executive_metric:.2f}")
```

### 2. Compute the Pareto frontier and identify the recommended configuration

```python
from agent_tco import (
    CostParams, GridSearch, WorkflowConfig,
    step_config_from_tier, TIER_B, TIER_C,
    pareto_frontier, rank_by_executive_metric,
)

cost_params = CostParams(labor_rate=60.0, review_time_hours=0.25, manual_process_cost=50.0)

j3 = WorkflowConfig("j3", [
    step_config_from_tier(k, TIER_C, acr=0.78 if k == 2 else 0.92, hitl_prob=0.20 if k == 4 else 0.05)
    for k in range(1, 6)
])
j9 = WorkflowConfig("j9", [
    step_config_from_tier(k, TIER_B if k == 2 else TIER_C,
                          acr=0.88 if k == 2 else 0.92,
                          hitl_prob=0.08 if k == 2 else (0.20 if k == 4 else 0.05))
    for k in range(1, 6)
])

results = GridSearch(cost_params=cost_params, n_runs=300, seed=42).evaluate_all([j3, j9])
frontier = pareto_frontier(results)
best = rank_by_executive_metric(frontier)[0]
print(f"Recommended: {best.config.config_id} — ${best.executive_metric:.2f} per autonomous outcome")
```

---

## Interpreting the Executive Metric

The executive metric `M = E[TCO] / ACR` is the expected cost per autonomously delivered outcome. When `M < C_manual` (the cost of the manual process), the agent is economically viable at this configuration. If `M > C_manual` for every configuration on the Pareto frontier, no configuration is cost-effective and the business case does not hold.

---

## Using a real LLM executor

The `StepExecutor` protocol accepts any callable `(step, context) -> (completed, hitl_triggered)`:

```python
from agent_tco import (
    CostParams, WorkflowConfig, WorkflowSimulator,
    step_config_from_tier, TIER_B, TIER_C,
)

cost_params = CostParams(labor_rate=60.0, review_time_hours=0.25, manual_process_cost=50.0)

j9 = WorkflowConfig("j9", [
    step_config_from_tier(k, TIER_B if k == 2 else TIER_C,
                          acr=0.88 if k == 2 else 0.92,
                          hitl_prob=0.08 if k == 2 else (0.20 if k == 4 else 0.05))
    for k in range(1, 6)
])

def my_llm_executor(step, context):
    # Replace with your real LLM call.
    # Return (completed: bool, hitl_triggered: bool).
    completed = True
    hitl_triggered = False
    return completed, hitl_triggered

sim = WorkflowSimulator(cost_params=cost_params, executor=my_llm_executor)
run = sim.simulate_run(j9)
print(run.scenario, run.total_cost)
```

In tests, replace `my_llm_executor` with `Mock(return_value=(True, False))`.

---

## API Reference

See [API.md](API.md).

---

## License

MIT License. See [LICENSE](LICENSE).
