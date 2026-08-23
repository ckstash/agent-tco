"""Visualization utilities for agent-tco grid search results.

All functions return a matplotlib Figure object so the caller controls
display and saving.

Example::

    from agent_tco.plot import plot_pareto_frontier, plot_sensitivity

    fig = plot_pareto_frontier(results, manual_process_cost=4.50)
    fig.savefig("pareto.pdf")

    fig2 = plot_sensitivity(baseline_config, step_labels={1: "Extract", 2: "Classify"})
    fig2.savefig("sensitivity.pdf")
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_tco.grid import GridResult
    from agent_tco.model import WorkflowConfig


def plot_pareto_frontier(
    results: list["GridResult"],
    manual_process_cost: float | None = None,
    label_frontier: bool = True,
    style: str = "default",
    figsize: tuple[float, float] = (7.0, 5.5),
    title: str = "",
    x_min: float = 0.0,
    x_max: float | None = None,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> "matplotlib.figure.Figure":
    """Plot the ACR vs. E[TCO] Pareto frontier.

    Args:
        results: list of GridResult objects (pareto_frontier() should be called first).
        manual_process_cost: if provided, draws a mu = C_manual iso-metric line.
        label_frontier: annotate Pareto-optimal points with their config_id.
        style: matplotlib style string. Use "paper" for publication-quality serif output.
        figsize: figure size in inches.
        title: axes title.
        x_min: left bound of the x-axis (E[TCO]). Default 0.0.
        x_max: right bound of the x-axis. Default None (auto: 15% beyond max E[TCO]).
        y_min: lower bound of the y-axis (ACR). Default 0.0. Pass a value > 0
               to zoom in; broken-axis (//) markers are shown automatically.
        y_max: upper bound of the y-axis. Default 1.05.

    Returns:
        matplotlib Figure.

    Examples::

        # Default full-scale view
        fig = plot_pareto_frontier(results)

        # Zoomed to spread out Pareto-optimal points — // marks appear automatically
        fig = plot_pareto_frontier(results, y_min=0.6, y_max=1.0)
    """

    if style == "paper":
        matplotlib.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })

    fig, ax = plt.subplots(figsize=figsize)

    frontier = [r for r in results if not r.pareto_dominated]
    dominated = [r for r in results if r.pareto_dominated]

    # Dominated points
    if dominated:
        ax.scatter(
            [r.expected_tco for r in dominated],
            [r.acr for r in dominated],
            color="gray", s=60, facecolors="white", edgecolors="gray",
            linewidths=1.2, zorder=4, label="Dominated",
        )

    # Pareto-optimal points
    if frontier:
        ax.scatter(
            [r.expected_tco for r in frontier],
            [r.acr for r in frontier],
            color="steelblue", s=120, zorder=5, label="Pareto-optimal",
        )

    # Connect frontier points
    if len(frontier) > 1:
        fp = sorted(frontier, key=lambda r: r.expected_tco)
        ax.plot(
            [r.expected_tco for r in fp],
            [r.acr for r in fp],
            color="steelblue", linewidth=2, zorder=3,
        )

    # Label frontier points
    if label_frontier and frontier:
        for r in frontier:
            ax.annotate(
                r.config.config_id,
                xy=(r.expected_tco, r.acr),
                xytext=(r.expected_tco + 0.05, r.acr + 0.008),
                fontsize=8, color="steelblue", fontweight="bold",
            )

    # Iso-metric lines (mu = E[TCO]/ACR = constant) — one legend entry for all
    all_tco = [r.expected_tco for r in results]
    _x_iso_min = max(0.0, min(all_tco) * 0.85)
    _x_iso_max = max(all_tco) * 1.15
    x_range = np.linspace(_x_iso_min, _x_iso_max, 300)
    _iso_label_added = False
    for r in frontier:
        mu = r.executive_metric
        if mu < float("inf"):
            label = "Cost per outcome" if not _iso_label_added else None
            ax.plot(x_range, x_range / mu, color="gray", linewidth=0.9,
                    linestyle=":", zorder=2, alpha=0.5, label=label)
            _iso_label_added = True

    # C_manual viability line
    if manual_process_cost is not None:
        ax.plot(x_range, x_range / manual_process_cost, color="red",
                linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"Manual process cost (${manual_process_cost:.2f})")

    ax.set_xlabel("Expected TCO per run ($\\$$)")
    ax.set_ylabel("Autonomous Completion Rate")
    if title:
        ax.set_title(title)

    _x_min = x_min if x_min != 0.0 else max(0.0, min(all_tco) * 0.85)
    _x_max = x_max if x_max is not None else max(all_tco) * 1.15
    _y_max = min(y_max, 1.0)
    ax.set_xlim(left=_x_min, right=_x_max)
    ax.set_ylim(bottom=y_min, top=_y_max)

    if y_min > 0:
        dx, dy = 0.012, 0.018
        for yb in (0.0, 0.012):
            ax.plot([-dx, dx], [yb - dy, yb + dy], color="black", lw=1.0,
                    clip_on=False, transform=ax.transAxes, zorder=10)

    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_sensitivity(
    config: "WorkflowConfig",
    step_labels: dict[int, str] | None = None,
    style: str = "default",
    figsize: tuple[float, float] = (7.5, 4.2),
    color: str = "steelblue",
    y_min: float = 0.0,
    y_max: float | None = None,
    x_min: float = 0.0,
    x_max: float | None = None,
) -> "matplotlib.figure.Figure":
    """Plot the critical step sensitivity bar chart.

    Args:
        config: baseline WorkflowConfig (typically all-cheapest-tier).
        step_labels: optional dict mapping step_index -> human-readable label.
        style: matplotlib style. Use "paper" for publication-quality output.
        figsize: figure size in inches.
        color: bar color.
        y_min: lower bound of the y-axis. Default 0.0. Pass a value > 0 to
               zoom in; broken-axis (//) markers are shown automatically.
        y_max: upper bound of the y-axis. Default None (auto: 8% above max bar).
        x_min: left margin before the first bar. Default 0.0.
        x_max: right margin after the last bar. Default None (auto).

    Returns:
        matplotlib Figure.

    Examples::

        # Default: full scale from 0
        fig = plot_sensitivity(baseline)

        # Zoomed to show spread — // marks appear automatically
        fig = plot_sensitivity(baseline, y_min=0.5)

        # Full axis control
        fig = plot_sensitivity(baseline, y_min=0.5, y_max=0.8)
    """

    from agent_tco.sensitivity import critical_step_sensitivity

    if style == "paper":
        matplotlib.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })

    ranking = critical_step_sensitivity(config)
    # Sort by step_index for display
    ranking_by_idx = sorted(ranking, key=lambda x: x[0])
    indices = [r[0] for r in ranking_by_idx]
    values = [r[1] for r in ranking_by_idx]
    labels = [(step_labels or {}).get(i, f"Step {i}") for i in indices]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color=color, edgecolor=color)
    ax.set_ylabel("ACR sensitivity to step accuracy")
    ax.set_ylim(bottom=y_min, top=y_max if y_max is not None else max(values) * 1.08)
    if x_min != 0.0 or x_max is not None:
        n = len(labels)
        ax.set_xlim(left=x_min - 0.5, right=(x_max if x_max is not None else n - 0.5))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    if y_min > 0:
        # Show broken-axis (//) markers to signal that y does not start at zero
        dx, dy = 0.012, 0.018
        for yb in (0.0, 0.012):
            ax.plot([-dx, dx], [yb - dy, yb + dy], color="black", lw=1.0,
                    clip_on=False, transform=ax.transAxes, zorder=10)

    fig.tight_layout()
    return fig


def plot_scenario_breakdown(
    results: list["GridResult"],
    style: str = "default",
    figsize: tuple[float, float] = (10, 5),
) -> "matplotlib.figure.Figure":
    """Plot scenario probability breakdown as a stacked bar chart.

    Shows P(i1)..P(i6) for each configuration, sorted by executive metric.

    Returns:
        matplotlib Figure.
    """


    from agent_tco.model import Scenario

    if style == "paper":
        matplotlib.rcParams.update({"font.family": "serif", "font.size": 9})

    sorted_results = sorted(results, key=lambda r: r.executive_metric)
    labels = [r.config.config_id for r in sorted_results]

    scenario_colors = {
        Scenario.HAPPY_PATH: "#4CAF50",
        Scenario.RETRY_RESOLVED: "#8BC34A",
        Scenario.HITL_SUCCESS: "#FFC107",
        Scenario.HITL_PARTIAL: "#FF9800",
        Scenario.COMPOUND_RECOVERY: "#FF5722",
        Scenario.ABORTED: "#F44336",
    }
    scenario_labels = {
        Scenario.HAPPY_PATH: "Happy path",
        Scenario.RETRY_RESOLVED: "Retry-resolved",
        Scenario.HITL_SUCCESS: "HITL success",
        Scenario.HITL_PARTIAL: "HITL partial",
        Scenario.COMPOUND_RECOVERY: "Compound recovery",
        Scenario.ABORTED: "Aborted",
    }

    x = np.arange(len(sorted_results))
    fig, ax = plt.subplots(figsize=figsize)

    bottoms = np.zeros(len(sorted_results))
    for scenario in Scenario:
        vals = np.array([
            r.scenario_stats[scenario].probability for r in sorted_results
        ])
        ax.bar(x, vals, bottom=bottoms, color=scenario_colors[scenario],
               label=scenario_labels[scenario], width=0.6)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              fontsize=7, ncol=3, framealpha=0.9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    return fig
