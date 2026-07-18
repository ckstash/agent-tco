from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Scenario(Enum):
    """The six mutually exclusive trajectory outcome classes."""

    HAPPY_PATH = "i1"          # r=0, h=0, s=success
    RETRY_RESOLVED = "i2"      # r>=1, h=0, s=success
    HITL_SUCCESS = "i3"        # h=1, r=0, s=success
    HITL_PARTIAL = "i4"        # h=1, s=partial
    COMPOUND_RECOVERY = "i5"   # r>=1, h=1, s=success
    ABORTED = "i6"             # s=aborted (any r, h)


@dataclass
class StepConfig:
    """Per-step configuration for one LLM touchpoint in a workflow."""

    step_index: int        # 1-based position in the workflow
    model_tier: str        # informational label, e.g. "A", "B", "C"
    acr: float             # per-step autonomous completion rate in [0, 1]
    hitl_prob: float       # probability of HITL trigger in [0, 1]
    inference_cost: float  # cost per LLM call for this step (dollars)


@dataclass
class WorkflowConfig:
    """A complete workflow configuration: one StepConfig per touchpoint."""

    config_id: str
    steps: list[StepConfig]

    @property
    def n_steps(self) -> int:
        """Return the number of steps in this configuration."""
        return len(self.steps)


@dataclass
class CostParams:
    """Cost parameters that are independent of the workflow configuration."""

    labor_rate: float           # dollars per hour
    review_time_hours: float    # hours per HITL review event
    manual_process_cost: float  # opportunity cost applied when a run is aborted

    @property
    def hitl_cost_per_event(self) -> float:
        """Return the cost of one HITL review event in dollars."""
        return self.labor_rate * self.review_time_hours


@dataclass
class SimulationRun:
    """Result of one simulated workflow run."""

    scenario: Scenario
    total_cost: float
    retry_count: int
    hitl_triggered: bool
