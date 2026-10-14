from __future__ import annotations

from typing import Protocol

from agent_tco.model import StepConfig, StepOutcome


class StepExecutor(Protocol):
    """Protocol for executing one LLM step in a workflow.

    A callable returns one categorical StepOutcome. WorkflowSimulator also accepts
    legacy ``(completed, hitl_triggered)`` tuples for backward compatibility.

    In production, implement this with a real LLM call.
    In tests, pass a lambda or unittest.mock.Mock to control outcomes without
    any network call.
    """

    def __call__(
        self,
        step: StepConfig,
        context: dict,
    ) -> StepOutcome:
        """Execute the step and return its categorical outcome."""
        ...
