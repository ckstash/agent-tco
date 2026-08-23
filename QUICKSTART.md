# agent-tco Quickstart

## What this package does

`agent-tco` finds the cheapest AI model configuration for your agentic workflow that still meets your quality requirements. It accounts for the full cost of each run: inference, human review (HITL) labor, retries, and the opportunity cost of runs that abort and fall back to a manual process.

Per-step benchmarking (pick the cheapest model that passes a quality threshold at each step) systematically selects the wrong configuration. It misses two effects that dominate real production costs:

1. **Compound accuracy degradation**: a 5-step workflow where each step passes at 92% only completes autonomously 62% of the time.
2. **HITL cost amplification**: cheaper models trigger more human review escalations whose labor cost exceeds the inference savings.

`agent-tco` identifies the configuration on the Pareto frontier of Autonomous Completion Rate (ACR) vs. Expected Total Cost of Ownership (E[TCO]) that minimizes cost per autonomous outcome.

---

## Install

```bash
pip install agent-tco
```

Requires Python 3.12+.

---

## Five-minute example

Copy and run this. It uses simulation only — no LLM API calls required.

```python
from agent_tco import CostParams, GridSearch, WorkflowConfig, StepConfig
from agent_tco import pareto_frontier, rank_by_executive_metric
from agent_tco.plot import plot_pareto_frontier, plot_sensitivity
from agent_tco.pricing import TierCost

# --- 1. Define your cost parameters ---
# labor_rate: cost per hour for a human reviewer (from HR data)
# review_time_hours: minutes per review event, converted to hours
# manual_process_cost: cost to complete the task manually if the agent aborts
cost_params = CostParams(
    labor_rate=60.0,           # $60/hr
    review_time_hours=0.25,    # 15 minutes per review
    manual_process_cost=4.50,  # C_manual: cost of one manual run
)

# --- 2. Define your model tiers ---
# Replace these with your actual current API pricing
tier_a = TierCost("A", cost_per_1k_tokens=0.50)   # frontier-class
tier_b = TierCost("B", cost_per_1k_tokens=0.08)   # mid-range
tier_c = TierCost("C", cost_per_1k_tokens=0.003)  # lightweight

# --- 3. Define your configurations ---
# j3: cheapest — all Tier C at every step
j3 = WorkflowConfig("j3", steps=[
    StepConfig(1, "C", acr=0.93, hitl_prob=0.03, inference_cost=0.003),
    StepConfig(2, "C", acr=0.80, hitl_prob=0.04, inference_cost=0.003),  # bottleneck
    StepConfig(3, "C", acr=0.96, hitl_prob=0.01, inference_cost=0.003),
    StepConfig(4, "C", acr=0.89, hitl_prob=0.12, inference_cost=0.003),
    StepConfig(5, "C", acr=0.92, hitl_prob=0.03, inference_cost=0.003),
])

# j9: Proposition 1 recommendation — upgrade only the bottleneck step (step 2)
j9 = WorkflowConfig("j9", steps=[
    StepConfig(1, "C", acr=0.93, hitl_prob=0.03, inference_cost=0.003),
    StepConfig(2, "B", acr=0.90, hitl_prob=0.02, inference_cost=0.08),   # upgraded
    StepConfig(3, "C", acr=0.96, hitl_prob=0.01, inference_cost=0.003),
    StepConfig(4, "C", acr=0.89, hitl_prob=0.04, inference_cost=0.003),  # cascade benefit
    StepConfig(5, "C", acr=0.92, hitl_prob=0.03, inference_cost=0.003),
])

j1 = WorkflowConfig("j1", steps=[
    StepConfig(k, "A", acr=0.99, hitl_prob=0.01, inference_cost=0.50)
    for k in range(1, 6)
])

# --- 4. Run the grid search ---
gs = GridSearch(cost_params=cost_params, n_runs=300, seed=42)
results = gs.evaluate_all([j1, j3, j9])

# --- 5. Compute the Pareto frontier ---
frontier = pareto_frontier(results)
best = rank_by_executive_metric(frontier)[0]

print(f"Recommended: {best.config.config_id}")
print(f"  ACR = {best.acr:.3f}  (fraction of runs completing without human intervention)")
print(f"  E[TCO] = ${best.expected_tco:.2f}  (expected cost per run)")
print(f"  µ = ${best.executive_metric:.2f}  (cost per autonomous outcome)")
print(f"  C_manual = ${cost_params.manual_process_cost:.2f}  (manual process cost)")
print(f"  Viable: {best.executive_metric < cost_params.manual_process_cost}")

# --- 6. Visualize ---
fig = plot_pareto_frontier(results, manual_process_cost=cost_params.manual_process_cost)
fig.savefig("pareto.pdf")
print("Pareto frontier saved to pareto.pdf")
```

**Expected output:**
```
Recommended: j9
  ACR = 0.853
  E[TCO] = $2.14
  µ = $2.51
  C_manual = $4.50
  Viable: True
```

---

## Let Proposition 1 build the grid for you

For workflows with many steps and tier options, use `ConfigSpaceBuilder` to generate
the targeted configuration grid automatically. It runs Proposition 1 (critical step
sensitivity) on your baseline configuration and selects the most promising
single-step upgrades to simulate.

```python
from agent_tco.config_space import ConfigSpaceBuilder, TierOption
from agent_tco.pricing import TierCost

tier_a = TierCost("A", cost_per_1k_tokens=0.50)
tier_b = TierCost("B", cost_per_1k_tokens=0.08)
tier_c = TierCost("C", cost_per_1k_tokens=0.003)

builder = ConfigSpaceBuilder(steps=[
    TierOption(
        step_index=1, label="Extract",
        tiers=[tier_a, tier_b, tier_c],
        acr_by_tier={tier_a: 0.98, tier_b: 0.96, tier_c: 0.93},
        hitl_by_tier={tier_a: 0.01, tier_b: 0.02, tier_c: 0.03},
    ),
    TierOption(
        step_index=2, label="Classify",
        tiers=[tier_a, tier_b, tier_c],
        acr_by_tier={tier_a: 0.97, tier_b: 0.90, tier_c: 0.80},
        hitl_by_tier={tier_a: 0.01, tier_b: 0.02, tier_c: 0.04},
    ),
    # ... one TierOption per step
])

# Returns ~12 targeted configs: all-tier anchors + Proposition-1-screened upgrades
configs = builder.targeted_grid(k=12)

results = GridSearch(cost_params=cost_params, n_runs=300).evaluate_all(configs)
frontier = pareto_frontier(results)
```

---

## Where to get your parameters

| Parameter | Where to get it |
|---|---|
| `labor_rate` | HR data; fully-loaded cost per hour for your reviewer role |
| `review_time_hours` | Time-motion study or estimate; 10–20 minutes is typical |
| `manual_process_cost` | Process cost accounting; cost of one manual execution |
| `acr` | Prior deployment logs, existing benchmarks, or conservative estimate |
| `hitl_prob` | Prior logs or estimate; start with 0.05 for most steps, higher for validation steps |
| `inference_cost` | Provider pricing page; cost per 1K output tokens × avg tokens per call / 1000 |

If you do not have prior logs, start with conservative estimates (lower `acr`, higher `hitl_prob`) and treat the result as a lower bound on the business case.

---

## Save results and recompute under new pricing

Scenario probability estimates are independent of cost parameters. Save once, recompute the Pareto frontier whenever pricing changes.

```python
from agent_tco.persist import save_results, load_results
from agent_tco.pareto import pareto_frontier

# Save after grid search
save_results(results, "run_2026_08_01.json")

# Later, with updated pricing — no re-simulation needed
results = load_results("run_2026_08_01.json")
frontier = pareto_frontier(results)
```

---

## How to interpret the output

| Quantity | Meaning | Good value |
|---|---|---|
| **ACR** | Fraction of runs completing without human intervention | As high as possible |
| **E[TCO]** | Expected cost per run (inference + HITL labor + retry + abort) | As low as possible |
| **µ = E[TCO]/ACR** | Expected cost per autonomous outcome | µ < C_manual |
| **C_manual** | Cost of one manual execution | Your baseline |
| **Pareto-optimal** | No other config is both cheaper and more reliable | Your decision set |
| **Dominated** | At least one config beats it on both ACR and E[TCO] | Eliminate these |

A configuration is **viable** when µ < C_manual: the agent delivers outcomes more cheaply than the manual process. If every configuration on the Pareto frontier has µ > C_manual, no configuration is cost-effective at current parameters and the workflow design needs revisiting.

---

## Next steps

- [API reference](API.md)
- [Paper: Beyond Point Solutions — Cost-Quality Optimization of Multi-Model Agentic Workflows](https://arxiv.org/abs/XXXX)
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute
- [examples/financial_close.ipynb](examples/financial_close.ipynb) — full worked example
