from __future__ import annotations

from typing import Protocol

from agent_tco.model import StepConfig


class StepExecutor(Protocol):
    """Protocol for executing one LLM step in a workflow.

    Any callable matching (step: StepConfig, context: dict) -> tuple[bool, bool]
    satisfies this protocol. The return value is (completed, hitl_triggered).

    In production, implement this with a real LLM call.
    In tests, pass a lambda or unittest.mock.Mock to control outcomes without
    any network call.
    """

    def __call__(
        self,
        step: StepConfig,
        context: dict,
    ) -> tuple[bool, bool]:
        """Execute the step and return (completed, hitl_triggered)."""
        ...
