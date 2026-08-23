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

    def __post_init__(self) -> None:
        if not (0.0 <= self.acr <= 1.0):
            raise ValueError(f"acr must be in [0, 1]; got {self.acr} at step {self.step_index}")
        if not (0.0 <= self.hitl_prob <= 1.0):
            raise ValueError(f"hitl_prob must be in [0, 1]; got {self.hitl_prob} at step {self.step_index}")
        if self.inference_cost < 0:
            raise ValueError(f"inference_cost must be non-negative; got {self.inference_cost}")
        if self.step_index < 1:
            raise ValueError(f"step_index must be >= 1; got {self.step_index}")


@dataclass
class WorkflowConfig:
    """A complete workflow configuration: one StepConfig per touchpoint."""

    config_id: str
    steps: list[StepConfig]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("WorkflowConfig requires at least one step.")
        indices = [s.step_index for s in self.steps]
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate step indices in '{self.config_id}': {sorted(indices)}")
        expected = list(range(min(indices), max(indices) + 1))
        if sorted(indices) != expected:
            raise ValueError(
                f"Step indices in '{self.config_id}' must be contiguous; "
                f"got {sorted(indices)}, expected {expected}"
            )

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

    def __post_init__(self) -> None:
        if self.labor_rate < 0:
            raise ValueError(f"labor_rate must be non-negative; got {self.labor_rate}")
        if self.review_time_hours < 0:
            raise ValueError(f"review_time_hours must be non-negative; got {self.review_time_hours}")
        if self.manual_process_cost < 0:
            raise ValueError(f"manual_process_cost must be non-negative; got {self.manual_process_cost}")

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
