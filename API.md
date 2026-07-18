<a id="agent_tco"></a>

# agent\_tco

<a id="agent_tco.grid"></a>

# agent\_tco.grid

<a id="agent_tco.grid.ScenarioStats"></a>

## ScenarioStats Objects

```python
@dataclass
class ScenarioStats()
```

Statistics for one scenario class across N simulation runs.

<a id="agent_tco.grid.ScenarioStats.probability"></a>

#### probability

empirical P_i = count_i / N

<a id="agent_tco.grid.ScenarioStats.ci_lower"></a>

#### ci\_lower

95% Beta credible interval lower bound

<a id="agent_tco.grid.ScenarioStats.ci_upper"></a>

#### ci\_upper

95% Beta credible interval upper bound

<a id="agent_tco.grid.ScenarioStats.mean_cost"></a>

#### mean\_cost

empirical mean cost within this scenario (0.0 if no runs)

<a id="agent_tco.grid.GridResult"></a>

## GridResult Objects

```python
@dataclass
class GridResult()
```

Aggregated results for one WorkflowConfig after N simulation runs.

<a id="agent_tco.grid.GridResult.acr"></a>

#### acr

P(i1) + P(i2)

<a id="agent_tco.grid.GridResult.expected_tco"></a>

#### expected\_tco

sum_i P_i * mean_cost_i

<a id="agent_tco.grid.GridResult.executive_metric"></a>

#### executive\_metric

expected_tco / acr; inf if acr == 0

<a id="agent_tco.grid.GridSearch"></a>

## GridSearch Objects

```python
class GridSearch()
```

Evaluate agentic workflow configurations over N simulation runs each.

<a id="agent_tco.grid.GridSearch.__init__"></a>

#### \_\_init\_\_

```python
def __init__(cost_params: CostParams,
             n_runs: int = 300,
             seed: int = 42) -> None
```

Initialize the grid search with cost parameters and simulation budget.

<a id="agent_tco.grid.GridSearch.evaluate"></a>

#### evaluate

```python
def evaluate(config: WorkflowConfig) -> GridResult
```

Run n_runs simulations for one config and return a GridResult.

<a id="agent_tco.grid.GridSearch.evaluate_all"></a>

#### evaluate\_all

```python
def evaluate_all(configs: list[WorkflowConfig]) -> list[GridResult]
```

Evaluate all configs sequentially and return one GridResult per config.

<a id="agent_tco.pareto"></a>

# agent\_tco.pareto

<a id="agent_tco.pareto.dominates"></a>

#### dominates

```python
def dominates(a: GridResult, b: GridResult) -> bool
```

Return True if a Pareto-dominates b.

a dominates b when a.acr >= b.acr and a.expected_tco <= b.expected_tco,
with at least one strict inequality.

<a id="agent_tco.pareto.pareto_frontier"></a>

#### pareto\_frontier

```python
def pareto_frontier(results: list[GridResult]) -> list[GridResult]
```

Return the non-dominated subset of results.

Sets pareto_dominated=True on every dominated GridResult in-place.

<a id="agent_tco.pareto.executive_metric"></a>

#### executive\_metric

```python
def executive_metric(expected_tco: float, acr: float) -> float
```

Return expected_tco / acr; float('inf') if acr == 0.

<a id="agent_tco.pareto.rank_by_executive_metric"></a>

#### rank\_by\_executive\_metric

```python
def rank_by_executive_metric(results: list[GridResult]) -> list[GridResult]
```

Return results sorted ascending by executive_metric (lowest cost-per-outcome first).

<a id="agent_tco.sensitivity"></a>

# agent\_tco.sensitivity

<a id="agent_tco.sensitivity.marginal_acr_gain"></a>

#### marginal\_acr\_gain

```python
def marginal_acr_gain(config: WorkflowConfig, step_index: int) -> float
```

Return the marginal ACR gain from improving step step_index by one unit.

Implements Proposition 1: ∂ACR/∂q_k = product of q_j for all j != step_index.
step_index is 1-based. Uses the no-retry independence assumption.

<a id="agent_tco.sensitivity.critical_step_sensitivity"></a>

#### critical\_step\_sensitivity

```python
def critical_step_sensitivity(
        config: WorkflowConfig) -> list[tuple[int, float]]
```

Return (step_index, marginal_acr_gain) pairs sorted descending by gain.

The first entry is the bottleneck step — the highest-priority upgrade target.

<a id="agent_tco.sensitivity.sample_complexity"></a>

#### sample\_complexity

```python
def sample_complexity(c_max: float,
                      epsilon: float,
                      delta: float = 0.05) -> int
```

Return the minimum N for E[TCO] estimation within epsilon at confidence 1-delta.

Uses the Hoeffding bound from Proposition 2:
N >= ceil(c_max^2 * log(6 / delta) / (2 * epsilon^2))

<a id="agent_tco.model"></a>

# agent\_tco.model

<a id="agent_tco.model.Scenario"></a>

## Scenario Objects

```python
class Scenario(Enum)
```

The six mutually exclusive trajectory outcome classes (Lemma 1 of the paper).

<a id="agent_tco.model.Scenario.HAPPY_PATH"></a>

#### HAPPY\_PATH

r=0, h=0, s=success

<a id="agent_tco.model.Scenario.RETRY_RESOLVED"></a>

#### RETRY\_RESOLVED

r>=1, h=0, s=success

<a id="agent_tco.model.Scenario.HITL_SUCCESS"></a>

#### HITL\_SUCCESS

h=1, r=0, s=success

<a id="agent_tco.model.Scenario.HITL_PARTIAL"></a>

#### HITL\_PARTIAL

h=1, s=partial

<a id="agent_tco.model.Scenario.COMPOUND_RECOVERY"></a>

#### COMPOUND\_RECOVERY

r>=1, h=1, s=success

<a id="agent_tco.model.Scenario.ABORTED"></a>

#### ABORTED

s=aborted (any r, h)

<a id="agent_tco.model.StepConfig"></a>

## StepConfig Objects

```python
@dataclass
class StepConfig()
```

Per-step configuration for one LLM touchpoint in a workflow.

<a id="agent_tco.model.StepConfig.step_index"></a>

#### step\_index

1-based position in the workflow

<a id="agent_tco.model.StepConfig.model_tier"></a>

#### model\_tier

informational label, e.g. "A", "B", "C"

<a id="agent_tco.model.StepConfig.acr"></a>

#### acr

per-step autonomous completion rate in [0, 1]

<a id="agent_tco.model.StepConfig.hitl_prob"></a>

#### hitl\_prob

probability of HITL trigger in [0, 1]

<a id="agent_tco.model.StepConfig.inference_cost"></a>

#### inference\_cost

cost per LLM call for this step (dollars)

<a id="agent_tco.model.WorkflowConfig"></a>

## WorkflowConfig Objects

```python
@dataclass
class WorkflowConfig()
```

A complete workflow configuration: one StepConfig per touchpoint.

<a id="agent_tco.model.WorkflowConfig.n_steps"></a>

#### n\_steps

```python
@property
def n_steps() -> int
```

Return the number of steps in this configuration.

<a id="agent_tco.model.CostParams"></a>

## CostParams Objects

```python
@dataclass
class CostParams()
```

Cost parameters that are independent of the workflow configuration.

<a id="agent_tco.model.CostParams.labor_rate"></a>

#### labor\_rate

dollars per hour

<a id="agent_tco.model.CostParams.review_time_hours"></a>

#### review\_time\_hours

hours per HITL review event

<a id="agent_tco.model.CostParams.manual_process_cost"></a>

#### manual\_process\_cost

opportunity cost applied when a run is aborted

<a id="agent_tco.model.CostParams.hitl_cost_per_event"></a>

#### hitl\_cost\_per\_event

```python
@property
def hitl_cost_per_event() -> float
```

Return the cost of one HITL review event in dollars.

<a id="agent_tco.model.SimulationRun"></a>

## SimulationRun Objects

```python
@dataclass
class SimulationRun()
```

Result of one simulated workflow run.

<a id="agent_tco.pricing"></a>

# agent\_tco.pricing

<a id="agent_tco.pricing.TierCost"></a>

## TierCost Objects

```python
@dataclass
class TierCost()
```

Per-1K-output-token cost for one model tier.

<a id="agent_tco.pricing.TierCost.cost_per_1k_tokens"></a>

#### cost\_per\_1k\_tokens

dollars

<a id="agent_tco.pricing.TIER_A"></a>

#### TIER\_A

frontier model (e.g. GPT-4o / Claude Sonnet class)

<a id="agent_tco.pricing.TIER_B"></a>

#### TIER\_B

mid-range model (e.g. GPT-4o Mini / Claude Haiku class)

<a id="agent_tco.pricing.TIER_C"></a>

#### TIER\_C

lightweight model

<a id="agent_tco.pricing.inference_cost"></a>

#### inference\_cost

```python
def inference_cost(output_tokens: int, tier: TierCost) -> float
```

Return the inference cost in dollars for output_tokens at tier pricing.

<a id="agent_tco.pricing.step_config_from_tier"></a>

#### step\_config\_from\_tier

```python
def step_config_from_tier(step_index: int,
                          tier: TierCost,
                          acr: float,
                          hitl_prob: float,
                          avg_output_tokens: int = 500) -> StepConfig
```

Construct a StepConfig from a TierCost and estimated average token usage.

<a id="agent_tco.simulator"></a>

# agent\_tco.simulator

<a id="agent_tco.simulator.WorkflowSimulator"></a>

## WorkflowSimulator Objects

```python
class WorkflowSimulator()
```

Simulate one run of a workflow under a given configuration.

Implements the single-retry model: if a step does not complete on the first
attempt, one retry is made. If the retry also fails, the run is aborted
(scenario i6) and the opportunity cost is applied.

Pass an optional executor callable to drive steps with real LLM calls instead
of Bernoulli sampling. The executor must match StepExecutor protocol:
(step: StepConfig, context: dict) -> tuple[bool, bool] (completed, hitl_triggered).

<a id="agent_tco.simulator.WorkflowSimulator.__init__"></a>

#### \_\_init\_\_

```python
def __init__(cost_params: CostParams,
             seed: int = 42,
             executor: Callable | None = None) -> None
```

Initialize the simulator with cost parameters and optional LLM executor.

<a id="agent_tco.simulator.WorkflowSimulator.simulate_run"></a>

#### simulate\_run

```python
def simulate_run(config: WorkflowConfig, run_index: int = 0) -> SimulationRun
```

Simulate one workflow run and return the classified outcome.

<a id="agent_tco.simulator.WorkflowSimulator.classify_scenario"></a>

#### classify\_scenario

```python
def classify_scenario(retry_count: int, hitl_triggered: bool,
                      terminal_state: str) -> Scenario
```

Classify a run outcome into one of the six scenario classes (Lemma 1).

<a id="agent_tco.executor"></a>

# agent\_tco.executor

<a id="agent_tco.executor.StepExecutor"></a>

## StepExecutor Objects

```python
class StepExecutor(Protocol)
```

Protocol for executing one LLM step in a workflow.

Any callable matching (step: StepConfig, context: dict) -> tuple[bool, bool]
satisfies this protocol. The return value is (completed, hitl_triggered).

In production, implement this with a real LLM call.
In tests, pass a lambda or unittest.mock.Mock to control outcomes without
any network call.

<a id="agent_tco.executor.StepExecutor.__call__"></a>

#### \_\_call\_\_

```python
def __call__(step: StepConfig, context: dict) -> tuple[bool, bool]
```

Execute the step and return (completed, hitl_triggered).

