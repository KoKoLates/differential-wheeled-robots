"""Metrics computed purely from an :class:`Episode` log (doc §4.3).

Two families are reported separately and must not be mixed up:

* **time-indexed** (``pos_*``, ``heading_*``, ``along_*``, ``lateral_*``): error
  against the reference *at the same time* — schedule-aware trajectory tracking.
* **path** (``cte_*``, ``completion``): distance to the geometric path at the
  monotonic progress point — timing-free path following.

All metrics use ground truth; controllers never see these values.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from diffdrive.core.geometry import body_frame_error, wrap_angle
from diffdrive.references.path import ProgressTracker
from diffdrive.sim.episode import Episode

SETTLING_TOL = 0.05  # [m]
SUCCESS_COMPLETION = 0.99
SUCCESS_GOAL_TOL = 0.10  # [m]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if len(x) else math.nan


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else math.nan


def tracking_errors(ep: Episode) -> dict[str, np.ndarray]:
    """Per-physics-tick error signals (useful for plots as well as metrics)."""
    t = ep["t"]
    truth = ep["truth"]
    ref_pose, ref_u, _ = ep.reference.sample(t)
    pos = np.linalg.norm(truth[:, :2] - ref_pose[:, :2], axis=1)
    heading = wrap_angle(truth[:, 2] - ref_pose[:, 2])
    body = body_frame_error(truth, ref_pose) if len(t) else np.zeros((0, 3))

    path = ep.reference.path
    tracker = ProgressTracker(path)
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    v_max = float(ep.meta["scenario"]["command"]["v_max"])
    s = np.empty(len(t))
    cte = np.empty(len(t))
    for i, p in enumerate(truth):
        s[i] = tracker.update(p, 3.0 * v_max * dt + 0.05)
        cte[i] = path.signed_lateral_error(p, s[i])
    return {
        "t": t,
        "pos": pos,
        "heading": heading,
        "along": body[:, 0],
        "lateral": body[:, 1],
        "progress": s,
        "cte": cte,
        "ref_pose": ref_pose,
        "ref_control": ref_u,
    }


def compute_metrics(ep: Episode) -> dict[str, Any]:
    sc = ep.meta["scenario"]
    dt_control = float(sc["timing"]["dt_control"])
    cmd = sc["command"]
    out: dict[str, Any] = {"termination": ep.termination}
    t = ep["t"]
    if len(t) < 2:
        out["success"] = False
        return out
    dt = float(t[1] - t[0])
    e = tracking_errors(ep)

    # time-indexed tracking
    out["pos_rmse"] = _rms(e["pos"])
    out["pos_max"] = float(e["pos"].max())
    out["pos_final"] = float(e["pos"][-1])
    out["heading_rmse"] = _rms(e["heading"])
    out["heading_mae"] = float(np.mean(np.abs(e["heading"])))
    out["along_rmse"] = _rms(e["along"])
    out["lateral_rmse"] = _rms(e["lateral"])
    out["settling_time"] = _settling_time(t, e["pos"], SETTLING_TOL)

    # path following
    length = ep.reference.length
    out["cte_rmse"] = _rms(e["cte"])
    out["cte_max"] = float(np.abs(e["cte"]).max())
    out["completion"] = float(e["progress"][-1] / length) if length > 0 else math.nan
    goal = ep.reference.path.xy[-1]
    out["goal_distance"] = float(np.linalg.norm(ep["truth"][-1, :2] - goal))

    # control effort / smoothness (applied velocities, physics ticks)
    applied = ep["applied"]
    out["effort"] = float(np.sum(applied**2) * dt)
    out["effort_deviation"] = float(np.sum((applied - e["ref_control"]) ** 2) * dt)
    command = ep["command"]
    if len(command) > 1:
        du = np.diff(command, axis=0) / dt_control
        out["smoothness"] = float(np.mean(np.sum(du**2, axis=1)))
    else:
        out["smoothness"] = math.nan

    # constraints on the *raw* controller output
    raw = ep["raw"]
    viol = (
        np.maximum.reduce(
            [
                np.zeros(len(raw)),
                cmd["v_min"] - raw[:, 0],
                raw[:, 0] - cmd["v_max"],
                np.abs(raw[:, 1]) - cmd["omega_max"],
            ]
        )
        if len(raw)
        else np.zeros(0)
    )
    out["constraint_violations"] = int(np.sum(viol > 1e-9))
    out["constraint_violation_max"] = float(viol.max()) if len(viol) else 0.0

    # computation
    st = ep["solve_time"] * 1e3
    out["solve_ms_mean"] = float(np.mean(st)) if len(st) else math.nan
    out["solve_ms_p50"] = _pct(st, 50)
    out["solve_ms_p95"] = _pct(st, 95)
    out["solve_ms_p99"] = _pct(st, 99)
    out["solve_ms_max"] = float(np.max(st)) if len(st) else math.nan
    out["deadline_miss_rate"] = float(np.mean(st > dt_control * 1e3)) if len(st) else math.nan
    status = ep["status"]
    out["n_failed"] = int(np.sum(status == "failed"))
    out["n_fallback"] = int(np.sum(status == "fallback"))

    out["success"] = bool(
        ep.termination == "time_limit"
        and out["completion"] >= SUCCESS_COMPLETION
        and out["goal_distance"] <= SUCCESS_GOAL_TOL
        and out["n_failed"] == 0
    )
    return out


def _settling_time(t: np.ndarray, err: np.ndarray, tol: float) -> float:
    outside = np.nonzero(err > tol)[0]
    if len(outside) == 0:
        return 0.0
    last = outside[-1]
    return float(t[last + 1]) if last + 1 < len(t) else math.nan
