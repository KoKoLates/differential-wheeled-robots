"""Aggregate benchmark results across seeds."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

DEFAULT_METRICS = (
    "success",
    "pos_rmse",
    "cte_rmse",
    "heading_rmse",
    "effort_deviation",
    "solve_ms_p95",
)


def aggregate(
    rows: list[dict[str, Any]], metrics: tuple[str, ...] = DEFAULT_METRICS
) -> list[dict[str, Any]]:
    """Group rows by ``(scenario, controller)``; mean/std/n for each metric."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["scenario"]), str(r["controller"]))].append(r)
    out = []
    for (sc, ctrl), rs in groups.items():
        agg: dict[str, Any] = {"scenario": sc, "controller": ctrl, "n": len(rs)}
        for m in metrics:
            vals = np.array([_num(r.get(m)) for r in rs], dtype=float)
            vals = vals[~np.isnan(vals)]
            agg[f"{m}_mean"] = float(vals.mean()) if len(vals) else math.nan
            agg[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(agg)
    return out


def _num(v: Any) -> float:
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def format_table(agg: list[dict[str, Any]], metrics: tuple[str, ...] = DEFAULT_METRICS) -> str:
    """Markdown table (mean ± std across seeds)."""
    header = ["scenario", "controller", "n", *metrics]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for a in sorted(agg, key=lambda r: (r["scenario"], r["controller"])):
        cells = [a["scenario"], a["controller"], str(a["n"])]
        for m in metrics:
            mean, std = a[f"{m}_mean"], a[f"{m}_std"]
            if math.isnan(mean):
                cells.append("–")
            elif m == "success":
                cells.append(f"{100 * mean:.0f}%")
            else:
                cells.append(f"{mean:.4g} ± {std:.2g}" if a["n"] > 1 else f"{mean:.4g}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
