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

95% Wilson confidence interval lower bound

<a id="agent_tco.grid.ScenarioStats.ci_upper"></a>

#### ci\_upper

95% Wilson confidence interval upper bound

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

<a id="agent_tco.grid.GridResult.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Return a flat dict suitable for JSON serialization or DataFrame construction.

Scenario probabilities, Wilson confidence intervals, and mean costs are included
as separate columns (e.g. 'p_i1', 'ci_lo_i1', 'ci_hi_i1', 'cost_i1').

<a id="agent_tco.grid.GridResult.to_dataframe"></a>

#### to\_dataframe

```python
@staticmethod
def to_dataframe(results: list["GridResult"])
```

Convert a list of GridResults to a pandas DataFrame.

<a id="agent_tco.grid.recompute_tco"></a>

#### recompute\_tco

```python
def recompute_tco(results: list[GridResult],
                  new_cost_params: CostParams) -> list[GridResult]
```

Reprice stored review and fallback costs without re-running behavior.

Inference costs remain fixed because they belong to each ``StepConfig``.
Each scenario stores its mean review-event count, and aborted scenarios
contain exactly one manual-fallback charge. These statistics and the original
``cost_params`` are sufficient to adjust each scenario mean.

Legacy results without original cost parameters cannot be repriced safely.
Exact results retain point TCO intervals; Monte Carlo TCO intervals are
reset because their run-level cost samples are not stored.

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

<a id="agent_tco.grid.GridSearch.recommended_n_runs"></a>

#### recommended\_n\_runs

```python
@staticmethod
def recommended_n_runs(c_max: float,
                       epsilon: float = 0.05,
                       delta: float = 0.05) -> int
```

Return the minimum n_runs for E[TCO] estimation at target precision.

Uses the direct Hoeffding concentration bound. ``c_max`` must bound every
run cost; a typical or truncated cost does not preserve the guarantee.

**Example**:

  GridSearch.recommended_n_runs(c_max=0.10, epsilon=0.05)  # returns 8

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

<a id="agent_tco.grid.ParallelGridSearch"></a>

## ParallelGridSearch Objects

```python
class ParallelGridSearch(GridSearch)
```

Evaluate agentic workflow configurations concurrently.

Uses ThreadPoolExecutor for I/O-bound workloads (real LLM calls via executor)
and ProcessPoolExecutor for CPU-bound simulation-only workloads.

**Example**:

  gs = ParallelGridSearch(cost_params=cost_params, n_runs=300, max_workers=8)
  results = gs.evaluate_all(configs)

<a id="agent_tco.grid.ParallelGridSearch.__init__"></a>

#### \_\_init\_\_

```python
def __init__(cost_params: CostParams,
             n_runs: int = 300,
             seed: int = 42,
             max_workers: int = 4,
             use_threads: bool = True) -> None
```

**Arguments**:

- `max_workers` - number of parallel workers.
- `use_threads` - True (default) for ThreadPoolExecutor (I/O-bound LLM calls);
  False for ProcessPoolExecutor (CPU-bound simulation-only).

<a id="agent_tco.grid.ParallelGridSearch.evaluate_all"></a>

#### evaluate\_all

```python
def evaluate_all(configs: list[WorkflowConfig]) -> list[GridResult]
```

Evaluate all configs in parallel and return one GridResult per config.

<a id="agent_tco.plot"></a>

# agent\_tco.plot

Visualization utilities for agent-tco grid search results.

All functions return a matplotlib Figure object so the caller controls
display and saving.

Example::

from agent_tco.plot import plot_pareto_frontier, plot_sensitivity

fig = plot_pareto_frontier(results, manual_process_cost=4.50)
fig.savefig("pareto.pdf")

fig2 = plot_sensitivity(baseline_config, step_labels={1: "Extract", 2: "Classify"})
fig2.savefig("sensitivity.pdf")

<a id="agent_tco.plot.plot_measured_pareto_frontier"></a>

#### plot\_measured\_pareto\_frontier

```python
def plot_measured_pareto_frontier(
    results: list["MeasuredConfigResult"] | tuple["MeasuredConfigResult", ...],
    frontier: list["MeasuredConfigResult"] | tuple["MeasuredConfigResult", ...]
    | None = None,
    figsize: tuple[float, float] = (7.0, 5.5),
    title: str = "Measured configuration frontier"
) -> "matplotlib.figure.Figure"
```

Plot measured ACR/E[TCO] results with 95% marginal error bars.

<a id="agent_tco.plot.plot_pareto_frontier"></a>

#### plot\_pareto\_frontier

```python
def plot_pareto_frontier(results: list["GridResult"],
                         manual_process_cost: float | None = None,
                         label_frontier: bool = True,
                         style: str = "default",
                         figsize: tuple[float, float] = (7.0, 5.5),
                         title: str = "",
                         x_min: float = 0.0,
                         x_max: float | None = None,
                         y_min: float = 0.0,
                         y_max: float = 1.0) -> "matplotlib.figure.Figure"
```

Plot the ACR vs. E[TCO] Pareto frontier.

**Arguments**:

- `results` - list of GridResult objects (pareto_frontier() should be called first).
- `manual_process_cost` - if provided, draws a mu = C_manual iso-metric line.
- `label_frontier` - annotate Pareto-optimal points with their config_id.
- `style` - use ``"publication"`` for publication-quality serif output.
- `figsize` - figure size in inches.
- `title` - axes title.
- `x_min` - left bound of the x-axis (E[TCO]). Default 0.0.
- `x_max` - right bound of the x-axis. Default None (auto: 15% beyond max E[TCO]).
- `y_min` - lower bound of the y-axis (ACR). Default 0.0. Pass a value > 0
  to zoom in; broken-axis (//) markers are shown automatically.
- `y_max` - upper bound of the y-axis. Default 1.05.
  

**Returns**:

  matplotlib Figure.
  
  Examples::
  
  # Default full-scale view
  fig = plot_pareto_frontier(results)
  
  # Zoomed to spread out Pareto-optimal points — // marks appear automatically
  fig = plot_pareto_frontier(results, y_min=0.6, y_max=1.0)

<a id="agent_tco.plot.plot_sensitivity"></a>

#### plot\_sensitivity

```python
def plot_sensitivity(config: "WorkflowConfig",
                     step_labels: dict[int, str] | None = None,
                     style: str = "default",
                     figsize: tuple[float, float] = (7.5, 4.2),
                     color: str = "steelblue",
                     y_min: float = 0.0,
                     y_max: float | None = None,
                     x_min: float = 0.0,
                     x_max: float | None = None) -> "matplotlib.figure.Figure"
```

Plot the critical step sensitivity bar chart.

**Arguments**:

- `config` - baseline WorkflowConfig (typically all-cheapest-tier).
- `step_labels` - optional dict mapping step_index -> human-readable label.
- `style` - use ``"publication"`` for publication-quality output.
- `figsize` - figure size in inches.
- `color` - bar color.
- `y_min` - lower bound of the y-axis. Default 0.0. Pass a value > 0 to
  zoom in; broken-axis (//) markers are shown automatically.
- `y_max` - upper bound of the y-axis. Default None (auto: 8% above max bar).
- `x_min` - left margin before the first bar. Default 0.0.
- `x_max` - right margin after the last bar. Default None (auto).
  

**Returns**:

  matplotlib Figure.
  
  Examples::
  
  # Default: full scale from 0
  fig = plot_sensitivity(baseline)
  
  # Zoomed to show spread — // marks appear automatically
  fig = plot_sensitivity(baseline, y_min=0.5)
  
  # Full axis control
  fig = plot_sensitivity(baseline, y_min=0.5, y_max=0.8)

<a id="agent_tco.plot.plot_scenario_breakdown"></a>

#### plot\_scenario\_breakdown

```python
def plot_scenario_breakdown(
    results: list["GridResult"],
    style: str = "default",
    figsize: tuple[float, float] = (10, 5)
) -> "matplotlib.figure.Figure"
```

Plot scenario probability breakdown as a stacked bar chart.

Shows P(i1)..P(i6) for each configuration, sorted by executive metric.

**Returns**:

  matplotlib Figure.

<a id="agent_tco.topology_search"></a>

# agent\_tco.topology\_search

Assignment search for structured workflow topologies.

<a id="agent_tco.topology_search.SearchReport"></a>

## SearchReport Objects

```python
@dataclass(frozen=True)
class SearchReport()
```

Results and audit metadata for one assignment search.

<a id="agent_tco.topology_search.topology_pareto_frontier"></a>

#### topology\_pareto\_frontier

```python
def topology_pareto_frontier(
        results: Sequence[TopologyResult]) -> list[TopologyResult]
```

Return the two-objective frontier in O(n log n) time.

<a id="agent_tco.topology_search.exhaustive_topology_search"></a>

#### exhaustive\_topology\_search

```python
def exhaustive_topology_search(topology: TopologyNode,
                               options: Mapping[str, Sequence[StepConfig]],
                               cost_params: CostParams) -> SearchReport
```

Evaluate the complete Cartesian assignment space exactly.

<a id="agent_tco.topology_search.guided_topology_search"></a>

#### guided\_topology\_search

```python
def guided_topology_search(topology: TopologyNode,
                           options: Mapping[str, Sequence[StepConfig]],
                           cost_params: CostParams,
                           max_evaluations: int = 500) -> SearchReport
```

Return a deterministic budgeted frontier approximation.

The search starts from homogeneous anchors, ranks tasks by topology-aware
Birnbaum importance, and repeatedly evaluates one-coordinate neighbors of
the current frontier. It is intentionally heuristic and carries no global
frontier guarantee.

<a id="agent_tco.persist"></a>

# agent\_tco.persist

Persistence utilities for GridResult objects.

Scenario probability estimates are independent of review and fallback prices.
Save results after a grid search, then reprice those components and recompute
the Pareto frontier without re-simulating workflow behavior.

Example::

from agent_tco.persist import save_results, load_results
from agent_tco.grid import recompute_tco
from agent_tco.pareto import pareto_frontier

# Save after grid search
save_results(results, "run_2026_08_01.json")

# Later, with updated review or fallback prices
results = load_results("run_2026_08_01.json")
repriced = recompute_tco(results, updated_cost_params)
frontier = pareto_frontier(repriced)

<a id="agent_tco.persist.save_results"></a>

#### save\_results

```python
def save_results(results: list[GridResult], path: str | Path) -> None
```

Save a list of GridResults to a JSON file.

New files include the original cost parameters so review and fallback
prices can be changed later with ``recompute_tco``. Inference prices are
part of the workflow configuration and are not changed by repricing.

<a id="agent_tco.persist.load_results"></a>

#### load\_results

```python
def load_results(path: str | Path) -> list[GridResult]
```

Load a list of GridResults from a JSON file previously saved by save_results().

<a id="agent_tco.hybrid"></a>

# agent\_tco.hybrid

Facade selecting exact structured evaluation or approximate graph simulation.

<a id="agent_tco.hybrid.evaluate_hybrid"></a>

#### evaluate\_hybrid

```python
def evaluate_hybrid(workflow,
                    assignment,
                    cost_params: CostParams,
                    *,
                    config_id: str | None = None,
                    n_runs: int = 1_000,
                    seed: int = 42,
                    context_classes: Sequence[str | None] | None = None)
```

Use exact block composition when possible and DAG Monte Carlo otherwise.

<a id="agent_tco.dag"></a>

# agent\_tco.dag

Profile-based discrete-event simulation for finite agent execution graphs.

The simulator is a screening tool. It estimates ACR and expected task cost from
declared task/model profiles and execution policies; it does not replace measured
validation of shortlisted assignments in the deployed agent stack.

<a id="agent_tco.dag.JoinPolicy"></a>

## JoinPolicy Objects

```python
class JoinPolicy(Enum)
```

Condition under which incoming activations start a task.

<a id="agent_tco.dag.OutgoingPolicy"></a>

## OutgoingPolicy Objects

```python
class OutgoingPolicy(Enum)
```

How matching outgoing edges are activated.

<a id="agent_tco.dag.SuccessPolicy"></a>

## SuccessPolicy Objects

```python
class SuccessPolicy(Enum)
```

Condition under which declared success nodes complete the workflow.

<a id="agent_tco.dag.CancellationPolicy"></a>

## CancellationPolicy Objects

```python
class CancellationPolicy(Enum)
```

Billing approximation for a task cancelled by a first-success race.

<a id="agent_tco.dag.TaskProfile"></a>

## TaskProfile Objects

```python
@dataclass(frozen=True)
class TaskProfile()
```

Categorical behavior, cost, and duration for one task/model option.

<a id="agent_tco.dag.FailureDomain"></a>

## FailureDomain Objects

```python
@dataclass(frozen=True)
class FailureDomain()
```

A run-level common cause that forces member task attempts to fail.

<a id="agent_tco.dag.DAGNode"></a>

## DAGNode Objects

```python
@dataclass(frozen=True)
class DAGNode()
```

One executable task and its graph-control policies.

<a id="agent_tco.dag.DAGEdge"></a>

## DAGEdge Objects

```python
@dataclass(frozen=True)
class DAGEdge()
```

A conditional graph transition.

Route weights may be supplied globally through ``probability`` or by context
class. Probabilities are used only when the source has ``ROUTE`` policy.

<a id="agent_tco.dag.WorkflowGraph"></a>

## WorkflowGraph Objects

```python
@dataclass(frozen=True)
class WorkflowGraph()
```

A finite execution graph with explicit termination guards.

<a id="agent_tco.dag.NodeExecution"></a>

## NodeExecution Objects

```python
@dataclass(frozen=True)
class NodeExecution()
```

One completed or cancelled task attempt in a graph run.

<a id="agent_tco.dag.DAGRun"></a>

## DAGRun Objects

```python
@dataclass(frozen=True)
class DAGRun()
```

One simulated complete graph trajectory.

<a id="agent_tco.dag.DAGSimulationResult"></a>

## DAGSimulationResult Objects

```python
@dataclass(frozen=True)
class DAGSimulationResult()
```

Monte Carlo aggregate for one graph assignment.

<a id="agent_tco.dag.DAGSimulator"></a>

## DAGSimulator Objects

```python
class DAGSimulator()
```

Execute profile-based graph trajectories using a discrete-event queue.

<a id="agent_tco.paper"></a>

# agent\_tco.paper

Reproducible constructed inputs for the companion paper.

The profiles in this module reproduce the worked examples in the companion paper.  They
are constructed verification inputs, not estimates for a production deployment.

<a id="agent_tco.paper.build_paper_chain_config"></a>

#### build\_paper\_chain\_config

```python
def build_paper_chain_config(tiers: str | tuple[str, ...]) -> WorkflowConfig
```

Build one five-step financial-close assignment from its tier string.

The validation-step assisted-outcome profile depends on the classification
tier, matching the explicit inter-step link declared in the manuscript.

<a id="agent_tco.paper.paper_chain_configurations"></a>

#### paper\_chain\_configurations

```python
def paper_chain_configurations() -> list[WorkflowConfig]
```

Return all 3^5 assignments in deterministic Cartesian-product order.

<a id="agent_tco.paper.evaluate_paper_chain"></a>

#### evaluate\_paper\_chain

```python
def evaluate_paper_chain() -> list[GridResult]
```

Exactly evaluate all 243 assignments in the paper's chain example.

<a id="agent_tco.paper.paper_chain_screened_ids"></a>

#### paper\_chain\_screened\_ids

```python
def paper_chain_screened_ids() -> set[str]
```

Return the 13 homogeneous and single-upgrade assignments in the paper.

<a id="agent_tco.paper.build_paper_structured_topology"></a>

#### build\_paper\_structured\_topology

```python
def build_paper_structured_topology() -> WorkflowBlock
```

Build the paper's eight-touchpoint structured close-control workflow.

<a id="agent_tco.paper.paper_structured_options"></a>

#### paper\_structured\_options

```python
def paper_structured_options() -> dict[str, tuple[StepConfig, ...]]
```

Return the three model options for each structured-example task.

<a id="agent_tco.paper.evaluate_paper_structured_exhaustive"></a>

#### evaluate\_paper\_structured\_exhaustive

```python
def evaluate_paper_structured_exhaustive() -> SearchReport
```

Exactly evaluate all 6,561 structured-example assignments.

<a id="agent_tco.paper.evaluate_paper_structured_guided"></a>

#### evaluate\_paper\_structured\_guided

```python
def evaluate_paper_structured_guided(
        max_evaluations: int = PAPER_GUIDED_BUDGET) -> SearchReport
```

Run the paper's deterministic budgeted structured-assignment search.

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

Return the descriptive cost-effectiveness ratio E[TCO]/ACR.

<a id="agent_tco.pareto.rank_by_executive_metric"></a>

#### rank\_by\_executive\_metric

```python
def rank_by_executive_metric(results: list[GridResult]) -> list[GridResult]
```

Return results sorted by the descriptive cost-effectiveness ratio.

<a id="agent_tco.dag_io"></a>

# agent\_tco.dag\_io

Versioned JSON/dict interchange for profile-based execution graphs.

<a id="agent_tco.dag_io.graph_from_dict"></a>

#### graph\_from\_dict

```python
def graph_from_dict(data: Mapping[str, Any]) -> WorkflowGraph
```

Validate and construct a WorkflowGraph from a JSON-compatible mapping.

<a id="agent_tco.dag_io.load_graph"></a>

#### load\_graph

```python
def load_graph(path: str | Path) -> WorkflowGraph
```

Load a WorkflowGraph from a UTF-8 JSON file.

<a id="agent_tco.dag_io.profiles_from_dict"></a>

#### profiles\_from\_dict

```python
def profiles_from_dict(
        data: Mapping[str, Any]) -> dict[str, tuple[TaskProfile, ...]]
```

Construct per-node model options from a JSON-compatible mapping.

<a id="agent_tco.dag_io.load_profiles"></a>

#### load\_profiles

```python
def load_profiles(path: str | Path) -> dict[str, tuple[TaskProfile, ...]]
```

Load per-node TaskProfile options from a UTF-8 JSON file.

<a id="agent_tco.sensitivity"></a>

# agent\_tco.sensitivity

<a id="agent_tco.sensitivity.marginal_acr_gain"></a>

#### marginal\_acr\_gain

```python
def marginal_acr_gain(config: WorkflowConfig, step_index: int) -> float
```

Return the derivative of chain ACR with respect to one per-attempt rate.

Let u_k = 1-q_k-h_k-p_k be the per-attempt non-completion probability.
With one retry, a step passes autonomously with probability
a_k = q_k(1+u_k). Chain ACR is the product of the a_k values, so
dACR/dq_k = (2-2q_k-h_k-p_k) product_{j != k} a_j when h_k and p_k
are held fixed.

<a id="agent_tco.sensitivity.critical_step_sensitivity"></a>

#### critical\_step\_sensitivity

```python
def critical_step_sensitivity(
        config: WorkflowConfig) -> list[tuple[int, float]]
```

Return (step_index, marginal_acr_gain) pairs sorted descending by gain.

The first entry is the highest local ACR-sensitivity step. This ranking is a
screening heuristic because available upgrades have different costs and
finite probability increments.

<a id="agent_tco.sensitivity.sensitivity_report"></a>

#### sensitivity\_report

```python
def sensitivity_report(config: WorkflowConfig,
                       step_labels: dict[int, str] | None = None) -> str
```

Return a formatted sensitivity ranking table as a plain-text string.

**Arguments**:

- `config` - the baseline WorkflowConfig (typically all-Tier-C).
- `step_labels` - optional mapping of step_index -> human-readable label.
  
  Example output:
  Rank  Step  Label               Sensitivity
  ------------------------------------------------
  1     2  Classify accounts       0.4454  ← highest sensitivity
  2     4  Validate totals         0.2690
  3     5  Generate report         0.1960
  4     1  Extract trial balance   0.1806
  5     3  Compute totals          0.1199

<a id="agent_tco.sensitivity.sample_complexity"></a>

#### sample\_complexity

```python
def sample_complexity(c_max: float,
                      epsilon: float,
                      delta: float = 0.05) -> int
```

Return the minimum N for E[TCO] estimation within epsilon at confidence 1-delta.

Uses the direct Hoeffding bound for the sample mean:
N >= ceil(c_max^2 * log(2 / delta) / (2 * epsilon^2))

<a id="agent_tco.model"></a>

# agent\_tco.model

<a id="agent_tco.model.Scenario"></a>

## Scenario Objects

```python
class Scenario(Enum)
```

The six mutually exclusive trajectory outcome classes.

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

<a id="agent_tco.model.StepOutcome"></a>

## StepOutcome Objects

```python
class StepOutcome(Enum)
```

Mutually exclusive outcome of one step attempt.

<a id="agent_tco.model.StepConfig"></a>

## StepConfig Objects

```python
@dataclass
class StepConfig()
```

Categorical per-attempt model for one touchpoint in a sequential workflow.

``q`` denotes the probability of autonomous completion on one attempt.
``hitl_prob`` denotes the probability that a reviewer completes the step,
and ``partial_prob`` denotes the probability that review produces a partial
terminal outcome. The residual probability is a retryable non-completion.

<a id="agent_tco.model.StepConfig.step_index"></a>

#### step\_index

1-based position in the workflow

<a id="agent_tco.model.StepConfig.model_tier"></a>

#### model\_tier

informational label, e.g. "A", "B", "C"

<a id="agent_tco.model.StepConfig.q"></a>

#### q

autonomous-completion probability per attempt

<a id="agent_tco.model.StepConfig.hitl_prob"></a>

#### hitl\_prob

HITL-assisted step-completion probability

<a id="agent_tco.model.StepConfig.inference_cost"></a>

#### inference\_cost

cost per LLM call for this step (dollars)

<a id="agent_tco.model.StepConfig.partial_prob"></a>

#### partial\_prob

HITL-assisted partial terminal probability

<a id="agent_tco.model.StepConfig.incomplete_prob"></a>

#### incomplete\_prob

```python
@property
def incomplete_prob() -> float
```

Return the residual probability of retryable non-completion.

<a id="agent_tco.model.StepConfig.autonomous_pass_prob"></a>

#### autonomous\_pass\_prob

```python
@property
def autonomous_pass_prob() -> float
```

Return autonomous completion probability with one retry.

<a id="agent_tco.model.StepConfig.outcome_from_uniform"></a>

#### outcome\_from\_uniform

```python
def outcome_from_uniform(value: float) -> StepOutcome
```

Map a uniform draw in [0, 1) to one categorical attempt outcome.

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

<a id="agent_tco.hypotheses"></a>

# agent\_tco.hypotheses

Reproducible stylized computational tests for the five paper hypotheses.

These experiments test implications of the declared workflow model. They are not
field validation: profile parameters are constructed, and deployment claims require
measured trajectories from the target agent and workload.

<a id="agent_tco.hypotheses.run_stylized_hypothesis_tests"></a>

#### run\_stylized\_hypothesis\_tests

```python
def run_stylized_hypothesis_tests(n_runs: int = 50_000,
                                  seed: int = 20260924) -> dict[str, Any]
```

Run the five constructed hypothesis checks and return JSON-ready results.

**Arguments**:

- `n_runs` - Monte Carlo runs for the dependent-error comparison in H3.
- `seed` - Deterministic base seed for H3.

<a id="agent_tco.dag_search"></a>

# agent\_tco.dag\_search

Budgeted configuration screening for profile-based execution graphs.

<a id="agent_tco.dag_search.dag_pareto_frontier"></a>

#### dag\_pareto\_frontier

```python
def dag_pareto_frontier(
        results: Sequence[DAGSimulationResult]) -> list[DAGSimulationResult]
```

Return the estimated ACR/E[TCO] frontier over evaluated assignments.

<a id="agent_tco.dag_search.budgeted_dag_search"></a>

#### budgeted\_dag\_search

```python
def budgeted_dag_search(
        simulator: DAGSimulator,
        options: Mapping[str, Sequence[TaskProfile]],
        max_evaluations: int = 500,
        n_runs_per_assignment: int = 200,
        context_classes: Sequence[str | None] | None = None
) -> DAGSearchReport
```

Screen a finite model-assignment space under a hard evaluation budget.

Homogeneous anchors and single-task alternatives from the cheapest baseline
are evaluated first. The search then expands one-coordinate neighbors of its
current estimated frontier. It is deterministic under the simulator seed and
carries no global-frontier guarantee.

<a id="agent_tco.experiment"></a>

# agent\_tco.experiment

Measured production-like experiments for simulator-shortlisted assignments.

<a id="agent_tco.experiment.MeasuredRun"></a>

## MeasuredRun Objects

```python
@dataclass(frozen=True)
class MeasuredRun()
```

Observed outcome and cost from one real agent execution.

<a id="agent_tco.experiment.ProductionExecutor"></a>

## ProductionExecutor Objects

```python
class ProductionExecutor(Protocol)
```

Adapter that runs the real agent stack for one assignment and input case.

<a id="agent_tco.experiment.AsyncProductionExecutor"></a>

## AsyncProductionExecutor Objects

```python
class AsyncProductionExecutor(Protocol)
```

Asynchronous adapter for a real agent stack or remote orchestration API.

<a id="agent_tco.experiment.ProductionExperimentRunner"></a>

## ProductionExperimentRunner Objects

```python
class ProductionExperimentRunner()
```

Run shortlisted assignments against real representative agent inputs.

The runner randomizes a balanced assignment/case/repeat schedule. The supplied
executor owns production concerns such as authentication, rate limiting,
tracing, scoring, retries inside the real agent, and provider billing capture.

<a id="agent_tco.experiment.AsyncProductionExperimentRunner"></a>

## AsyncProductionExperimentRunner Objects

```python
class AsyncProductionExperimentRunner()
```

Run a balanced measured experiment with bounded asynchronous concurrency.

<a id="agent_tco.exact"></a>

# agent\_tco.exact

Exact evaluation for finite sequential workflows with one retry per step.

<a id="agent_tco.exact.evaluate_exact"></a>

#### evaluate\_exact

```python
def evaluate_exact(config: WorkflowConfig,
                   cost_params: CostParams) -> GridResult
```

Return exact scenario probabilities and expected costs for a chain.

The evaluator enumerates the finite categorical outcome tree implied by each
StepConfig. Exact probabilities replace sampling intervals, so each interval
endpoint equals its probability and ``n_runs`` is zero.

<a id="agent_tco.pricing"></a>

# agent\_tco.pricing

Tier cost definitions for the agent-tco cost model.

The TIER_DEFAULT_* constants (also aliased as TIER_A/B/C) are illustrative
starting-point values only. Actual API pricing changes frequently and varies
by provider, region, and contract. Always supply your current pricing by
constructing TierCost objects directly.

Example::

tier_a = TierCost("A", cost_per_1k_tokens=0.015)   # your actual rate
tier_c = TierCost("C", cost_per_1k_tokens=0.0002)

<a id="agent_tco.pricing.TierCost"></a>

## TierCost Objects

```python
@dataclass(frozen=True)
class TierCost()
```

Per-1K-output-token inference cost for one model tier.

<a id="agent_tco.pricing.TierCost.cost_per_1k_tokens"></a>

#### cost\_per\_1k\_tokens

dollars; always supply your actual rate

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
                          q: float,
                          hitl_prob: float,
                          avg_output_tokens: int = 500,
                          partial_prob: float = 0.0) -> StepConfig
```

Construct a StepConfig from a TierCost and estimated average token usage.

<a id="agent_tco.topology"></a>

# agent\_tco.topology

Exact compositional evaluation for structured agentic workflow topologies.

This module deliberately supports a bounded policy language rather than arbitrary
graph execution: series, eager AND, ordered fallback OR, eager OR, and bounded
retry. Component outcomes are conditionally independent under an assignment.

<a id="agent_tco.topology.BlockType"></a>

## BlockType Objects

```python
class BlockType(Enum)
```

Execution policies supported by the structured exact evaluator.

<a id="agent_tco.topology.TaskNode"></a>

## TaskNode Objects

```python
@dataclass(frozen=True)
class TaskNode()
```

A configurable task leaf identified independently of its model assignment.

<a id="agent_tco.topology.WorkflowBlock"></a>

## WorkflowBlock Objects

```python
@dataclass(frozen=True)
class WorkflowBlock()
```

A nested structured workflow block with explicit execution semantics.

<a id="agent_tco.topology.OutcomeMoment"></a>

## OutcomeMoment Objects

```python
@dataclass(frozen=True)
class OutcomeMoment()
```

Probability and probability-weighted cost for one execution state.

<a id="agent_tco.topology.TopologyResult"></a>

## TopologyResult Objects

```python
@dataclass(frozen=True)
class TopologyResult()
```

Exact ACR, expected cost, and scenario probabilities for one assignment.

<a id="agent_tco.topology.collect_task_ids"></a>

#### collect\_task\_ids

```python
def collect_task_ids(node: TopologyNode) -> tuple[str, ...]
```

Return unique task IDs in deterministic traversal order.

<a id="agent_tco.topology.evaluate_topology"></a>

#### evaluate\_topology

```python
def evaluate_topology(topology: TopologyNode,
                      assignment: Mapping[str, StepConfig],
                      cost_params: CostParams,
                      config_id: str | None = None) -> TopologyResult
```

Evaluate one assignment exactly under the structured graph semantics.

<a id="agent_tco.topology.topology_acr"></a>

#### topology\_acr

```python
def topology_acr(topology: TopologyNode,
                 assignment: Mapping[str, StepConfig]) -> float
```

Return autonomous reliability without review or fallback prices.

<a id="agent_tco.topology.birnbaum_importance"></a>

#### birnbaum\_importance

```python
def birnbaum_importance(topology: TopologyNode,
                        assignment: Mapping[str, StepConfig],
                        node_id: str) -> float
```

Return the finite binary-component Birnbaum importance of one task.

The selected task is replaced by an autonomous component with reliability
one and then zero. This is a structural screening score; finite model moves
must still be evaluated with their costs.

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

Each attempt has one categorical outcome. Autonomous and human-assisted
completion advance to the next step; assisted completion also records review
cost. Partial completion terminates the run. A non-completion receives one
retry and then aborts.

An optional executor can return a StepOutcome. Legacy ``(completed, hitl)``
tuples remain supported.

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

Classify a run outcome into one of the six scenario classes.

<a id="agent_tco.config_space"></a>

# agent\_tco.config\_space

Configuration space enumeration and targeted grid generation.

For a workflow with K steps and T tier options per step, the full configuration
space has T^K entries — 3^5 = 243 for 5 steps and 3 tiers, 5^10 ≈ 10 million
for 10 steps and 5 tiers. This module provides tools to enumerate the full
space and, more usefully, to generate the small targeted grid that the critical
step sensitivity ranking recommends simulating.

Example::

from agent_tco.config_space import ConfigSpaceBuilder, TierOption
from agent_tco.pricing import TierCost

tier_a = TierCost("A", 0.50)
tier_b = TierCost("B", 0.08)
tier_c = TierCost("C", 0.003)

builder = ConfigSpaceBuilder(steps=[
TierOption(
step_index=1, label="Extract",
tiers=[tier_a, tier_b, tier_c],
q_by_tier={tier_a: 0.98, tier_b: 0.96, tier_c: 0.93},
hitl_by_tier={tier_a: 0.01, tier_b: 0.02, tier_c: 0.03},
),
# ... one TierOption per step
])

# Sensitivity-screened targeted grid (typically 10–20 configs)
configs = builder.targeted_grid(k=12)

<a id="agent_tco.config_space.TierOption"></a>

## TierOption Objects

```python
@dataclass
class TierOption()
```

Per-step tier options for config space construction.

<a id="agent_tco.config_space.TierOption.step_config"></a>

#### step\_config

```python
def step_config(tier: TierCost) -> StepConfig
```

Return a StepConfig for this step at the given tier.

<a id="agent_tco.config_space.ConfigSpaceBuilder"></a>

## ConfigSpaceBuilder Objects

```python
class ConfigSpaceBuilder()
```

Build and enumerate workflow configuration spaces.

**Arguments**:

- `steps` - one TierOption per workflow step, in step_index order.

<a id="agent_tco.config_space.ConfigSpaceBuilder.enumerate"></a>

#### enumerate

```python
def enumerate() -> list[WorkflowConfig]
```

Return the full Cartesian product of tier assignments.

For K steps with T tiers each, returns T^K WorkflowConfigs.
For large spaces (>10,000 configs), prefer targeted_grid().

<a id="agent_tco.config_space.ConfigSpaceBuilder.baseline"></a>

#### baseline

```python
def baseline(tier_index: int = -1) -> WorkflowConfig
```

Return the all-same-tier baseline configuration.

**Arguments**:

- `tier_index` - index into each step's tier list. Default -1 = last tier
  (typically the cheapest/lowest quality).

<a id="agent_tco.config_space.ConfigSpaceBuilder.targeted_grid"></a>

#### targeted\_grid

```python
def targeted_grid(k: int | None = None) -> list[WorkflowConfig]
```

Return a targeted grid of configurations using critical step sensitivity screening.

Generates:
- One all-same-tier anchor per tier
- For each step × tier combination where the step is a bottleneck candidate:
upgrade that step to that tier, keep all others at the cheapest tier.

This produces all-tier anchors plus single-step upgrades at each position
× each non-cheapest tier.

**Arguments**:

- `k` - if provided, return only the top-k configs by sensitivity priority.
  If None, return all single-step-upgrade configs.

<a id="agent_tco.executor"></a>

# agent\_tco.executor

<a id="agent_tco.executor.StepExecutor"></a>

## StepExecutor Objects

```python
class StepExecutor(Protocol)
```

Protocol for executing one LLM step in a workflow.

A callable returns one categorical StepOutcome. WorkflowSimulator also accepts
legacy ``(completed, hitl_triggered)`` tuples for backward compatibility.

In production, implement this with a real LLM call.
In tests, pass a lambda or unittest.mock.Mock to control outcomes without
any network call.

<a id="agent_tco.executor.StepExecutor.__call__"></a>

#### \_\_call\_\_

```python
def __call__(step: StepConfig, context: dict) -> StepOutcome
```

Execute the step and return its categorical outcome.

