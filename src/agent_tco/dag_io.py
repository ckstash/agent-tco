"""Versioned JSON/dict interchange for profile-based execution graphs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agent_tco.dag import (
    CancellationPolicy,
    DAGEdge,
    DAGNode,
    FailureDomain,
    JoinPolicy,
    OutgoingPolicy,
    SuccessPolicy,
    TaskProfile,
    WorkflowGraph,
)
from agent_tco.model import StepOutcome


SCHEMA_VERSION = "1.0"


def graph_from_dict(data: Mapping[str, Any]) -> WorkflowGraph:
    """Validate and construct a WorkflowGraph from a JSON-compatible mapping."""
    version = str(data.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported graph schema_version: {version}")
    nodes = {
        raw["id"]: DAGNode(
            node_id=raw["id"],
            join_policy=JoinPolicy(raw.get("join", "all")),
            outgoing_policy=OutgoingPolicy(raw.get("outgoing", "fan_out")),
            max_attempts=int(raw.get("max_attempts", 1)),
            max_visits=int(raw.get("max_visits", 1)),
            race_group=raw.get("race_group"),
            cancellation_policy=CancellationPolicy(raw.get("cancellation", "full")),
        )
        for raw in data["nodes"]
    }
    edges = tuple(
        DAGEdge(
            source=raw["source"],
            target=raw["target"],
            on=tuple(StepOutcome(value) for value in raw.get(
                "on", ["autonomous", "hitl_success"]
            )),
            probability=float(raw.get("probability", 1.0)),
            context_probabilities={
                str(key): float(value)
                for key, value in raw.get("context_probabilities", {}).items()
            },
        )
        for raw in data.get("edges", [])
    )
    domains = tuple(
        FailureDomain(
            raw["id"], float(raw["probability"]), frozenset(raw["members"])
        )
        for raw in data.get("failure_domains", [])
    )
    return WorkflowGraph(
        graph_id=data["graph_id"],
        nodes=nodes,
        edges=edges,
        entry_nodes=tuple(data["entry_nodes"]),
        success_nodes=tuple(data["success_nodes"]),
        success_policy=SuccessPolicy(data.get("success_policy", "all")),
        failure_domains=domains,
        max_events=int(data.get("max_events", 10_000)),
        max_time_seconds=float(data.get("max_time_seconds", 86_400.0)),
    )


def load_graph(path: str | Path) -> WorkflowGraph:
    """Load a WorkflowGraph from a UTF-8 JSON file."""
    return graph_from_dict(json.loads(Path(path).read_text()))


def profiles_from_dict(data: Mapping[str, Any]) -> dict[str, tuple[TaskProfile, ...]]:
    """Construct per-node model options from a JSON-compatible mapping."""
    result = {}
    for node_id, raw_profiles in data.items():
        result[node_id] = tuple(
            TaskProfile(
                profile_id=raw["profile_id"],
                q=float(raw["q"]),
                hitl_prob=float(raw.get("hitl_prob", 0.0)),
                partial_prob=float(raw.get("partial_prob", 0.0)),
                invocation_cost=float(raw["invocation_cost"]),
                duration_seconds=float(raw.get("duration_seconds", 1.0)),
                duration_jitter=float(raw.get("duration_jitter", 0.0)),
                failure_domains=tuple(raw.get("failure_domains", [])),
            )
            for raw in raw_profiles
        )
    return result


def load_profiles(path: str | Path) -> dict[str, tuple[TaskProfile, ...]]:
    """Load per-node TaskProfile options from a UTF-8 JSON file."""
    return profiles_from_dict(json.loads(Path(path).read_text()))
