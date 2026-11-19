"""Deterministic, multi-rate closed-loop episode runner.

Time advances in integer physics ticks (``dt_physics``). At each tick ``k``
(time ``t = k·dt_physics``) events are processed in this fixed order:

1. **sensor sample** — on sensor ticks, measure the *current* truth; unless
   dropped, queue it for delivery at ``k + sensor_latency``.
2. **sensor delivery** — samples due at ``k`` become the latest observation
   (the newest ``t_sample`` wins).
3. **control** — on control ticks, build ``Observation`` + ``ReferenceWindow``,
   call ``controller.compute`` (wall time measured, *not* simulated), pass the
   raw command through the limiter and queue it for ``k + command_latency``.
4. **command delivery** — commands due at ``k`` become the eligible command.
5. **record** the tick, then **advance** the plant by ``dt_physics`` with the
   eligible command (actuator lag → applied velocity → kinematics).
6. **termination** checks (non-finite state, persistent path deviation, time).

With zero latency a command computed at tick ``k`` is applied during
``[t_k, t_k + dt_physics)``; an observation sampled at ``k`` is visible at ``k``.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from diffdrive.controllers.base import Controller, ControllerContext
from diffdrive.core.geometry import compose_body_offset
from diffdrive.core.rng import RngStreams
from diffdrive.core.types import (
    Control,
    ControlResult,
    Observation,
    Pose2D,
    ReferenceWindow,
    RefRequest,
    Status,
)
from diffdrive.models.unicycle import Unicycle
from diffdrive.references.path import ProgressTracker
from diffdrive.references.trajectory import Trajectory, check_feasibility
from diffdrive.sim.components import CommandPipeline, PoseSensor, SensorSample, UnicyclePlant
from diffdrive.sim.episode import Episode
from diffdrive.sim.scenario import Scenario, ticks

DEFAULT_PATH_DS = 0.05


@dataclass(frozen=True)
class EpisodeStart:
    scenario: Scenario
    reference: Trajectory
    controller: Controller
    seed: int
    duration: float


@dataclass(frozen=True)
class ControlStep:
    t: float
    truth: np.ndarray
    observation: Observation
    reference: ReferenceWindow
    result: ControlResult
    command: Control


class Observer(Protocol):
    """Optional per-step hooks (progress bars, DAgger labelling). Return ``True`` from
    ``on_control`` to stop the episode early (e.g. the window was closed)."""

    def on_start(self, info: EpisodeStart) -> None: ...
    def on_control(self, step: ControlStep) -> bool | None: ...
    def on_end(self, episode: Episode) -> None: ...


class ControllerError(RuntimeError):
    pass


def run_episode(
    scenario: Scenario,
    controller: Controller,
    seed: int = 0,
    observers: Sequence[Observer] = (),
    label: str | None = None,
    raise_errors: bool = True,
    nominal_integrator: str | None = None,
) -> Episode:
    """Run one closed-loop episode and return its log.

    Args:
        scenario: the frozen experimental conditions.
        controller: any :class:`Controller`; it is ``reset`` here.
        seed: scenario seed; all exogenous randomness derives from it, so two
            controllers run with the same seed see identical noise.
        observers: optional per-step hooks.
        label: display name stored in the manifest (defaults to controller name).
        raise_errors: re-raise controller exceptions (``False`` records them as
            ``termination = "controller_error"``; used by benchmark suites).
        nominal_integrator: integrator of the model handed to the controller
            (defaults to the plant's integrator, i.e. no integrator mismatch).
    """
    sc = scenario
    tm = sc.timing
    dt = tm.dt_physics
    n_ctrl = ticks(tm.dt_control, dt, "dt_control")
    n_sens = ticks(sc.sensor_period, dt, "sensor.period")
    lat_s = ticks(sc.sensor.latency, dt, "sensor.latency", allow_zero=True)
    lat_u = ticks(sc.command.latency, dt, "command.latency", allow_zero=True)

    traj = sc.build_reference()
    limits = sc.limits()
    feas = check_feasibility(traj, limits)
    if not feas.ok:
        raise ValueError(f"scenario {sc.name!r}: infeasible reference: {feas.violations}")
    duration = sc.duration(traj)
    n_ticks = math.ceil(duration / dt - 1e-9)

    rng = RngStreams(seed)
    plant = UnicyclePlant(sc.plant, rng.get("plant"))
    ref_start = traj.sample(0.0)[0][0]
    ie = sc.initial_error
    plant.reset(compose_body_offset(ref_start, ie.dx, ie.dy, ie.dtheta))
    sensor = PoseSensor(
        sc.sensor, n_sens, lat_s, rng.get("sensor.noise"), rng.get("sensor.dropout")
    )
    pipeline = CommandPipeline(limits, tm.dt_control, lat_u)

    ctrl_seed = rng.child_seed("controller")
    ctx = ControllerContext(
        dt=tm.dt_control,
        limits=limits,
        model=Unicycle(nominal_integrator or sc.plant.integrator),
        rng=np.random.default_rng(ctrl_seed),
        seed=ctrl_seed,
    )
    controller.reset(ctx)
    req = controller.reference_request()
    if req.mode not in ("time", "path") or req.horizon < 0:
        raise ValueError(f"invalid reference request {req}")
    ref_dt = req.dt if req.dt is not None else tm.dt_control
    ref_ds = req.ds if req.ds is not None else DEFAULT_PATH_DS

    # Initial localisation: a (noisy) fix of the start pose is known at t=0.
    latest: SensorSample = sensor.measure(plant.pose, 0, 0.0)
    obs_progress = ProgressTracker(traj.path)
    eval_progress = ProgressTracker(traj.path)  # privileged; used only for termination

    info = EpisodeStart(sc, traj, controller, seed, duration)
    for ob in observers:
        ob.on_start(info)

    rec = _Recorder()
    termination = "time_limit"
    error_msg = ""
    deviation_since: float | None = None
    last_cmd = Control.zero()
    max_adv = 3.0 * limits.v_max * tm.dt_control + 0.2

    for k in range(n_ticks):
        t = k * dt

        # 1–2: sensing
        if sensor.is_sample_tick(k):
            measured, dropped = sensor.sample(plant.pose, k, t)
            rec.sensor_tick(t, measured.pose, dropped)
        for sample in sensor.deliver(k):
            if sample.t_sample >= latest.t_sample:
                latest = sample

        # 3: control
        if k % n_ctrl == 0:
            obs = Observation(Pose2D.from_array(latest.pose), t, latest.t_sample, last_cmd)
            if req.mode == "time":
                window = traj.window_time(t, req.horizon, ref_dt)
            else:
                s0 = obs_progress.update(latest.pose, max_adv)
                window = traj.window_path(s0, req.horizon, ref_ds)
            t0 = time.perf_counter()
            try:
                result = controller.compute(obs, window)
            except Exception as exc:
                if raise_errors:
                    raise
                termination, error_msg = "controller_error", f"{type(exc).__name__}: {exc}"
                break
            solve = time.perf_counter() - t0
            if not isinstance(result, ControlResult):
                raise ControllerError(
                    f"{type(controller).__name__}.compute must return ControlResult, "
                    f"got {type(result).__name__}"
                )
            raw = result.command
            if not (math.isfinite(raw.v) and math.isfinite(raw.omega)):
                termination = "numerical_failure"
                rec.control(t, obs, window, result, raw, Control.zero(), solve)
                break
            cmd = pipeline.submit(raw, k)
            last_cmd = cmd
            rec.control(t, obs, window, result, raw, cmd, solve)
            step = ControlStep(t, plant.pose.copy(), obs, window, result, cmd)
            if any(ob.on_control(step) for ob in observers):
                termination = "stopped"
                break

        # 4–5: actuation and physics
        eligible = pipeline.update(k)
        truth = plant.pose.copy()
        applied = plant.advance(eligible, dt)
        rec.tick(t, truth, eligible, applied)

        # 6: termination
        if not np.all(np.isfinite(plant.pose)):
            termination = "numerical_failure"
            break
        s_true = eval_progress.update(plant.pose, max_adv)
        dev = float(np.linalg.norm(plant.pose[:2] - traj.path.sample(s_true)[0][0]))
        if dev > sc.termination.max_path_deviation:
            deviation_since = t if deviation_since is None else deviation_since
            if t - deviation_since >= sc.termination.deviation_grace:
                termination = "tracking_failure"
                break
        else:
            deviation_since = None

    meta = _manifest(sc, controller, seed, req, label, termination, error_msg, duration)
    episode = Episode(meta, rec.arrays(), traj)
    for ob in observers:
        ob.on_end(episode)
    return episode


# ------------------------------------------------------------------ recorder
class _Recorder:
    def __init__(self) -> None:
        self.t: list[float] = []
        self.truth: list[np.ndarray] = []
        self.eligible: list[np.ndarray] = []
        self.applied: list[np.ndarray] = []
        self.c: dict[str, list[Any]] = {
            k: []
            for k in (
                "ctrl_t",
                "obs_pose",
                "obs_t_sample",
                "ref_pose",
                "ref_control",
                "ref_s",
                "raw",
                "command",
                "solve_time",
                "status",
                "pred_states",
                "pred_controls",
                "diag",
            )
        }
        self.sens_t: list[float] = []
        self.sens_pose: list[np.ndarray] = []
        self.sens_dropped: list[bool] = []

    def sensor_tick(self, t: float, pose: np.ndarray, dropped: bool) -> None:
        self.sens_t.append(t)
        self.sens_pose.append(pose)
        self.sens_dropped.append(dropped)

    def tick(self, t: float, truth: np.ndarray, eligible: Control, applied: np.ndarray) -> None:
        self.t.append(t)
        self.truth.append(truth)
        self.eligible.append(eligible.as_array())
        self.applied.append(applied)

    def control(
        self,
        t: float,
        obs: Observation,
        ref: ReferenceWindow,
        res: ControlResult,
        raw: Control,
        cmd: Control,
        solve: float,
    ) -> None:
        c = self.c
        c["ctrl_t"].append(t)
        c["obs_pose"].append(obs.pose.as_array())
        c["obs_t_sample"].append(obs.t_sample)
        c["ref_pose"].append(ref.poses[0])
        c["ref_control"].append(ref.controls[0])
        c["ref_s"].append(ref.s[0])
        c["raw"].append(raw.as_array())
        c["command"].append(cmd.as_array())
        c["solve_time"].append(solve)
        c["status"].append(res.status.value if isinstance(res.status, Status) else str(res.status))
        c["pred_states"].append(res.predicted_states)
        c["pred_controls"].append(res.predicted_controls)
        c["diag"].append(dict(res.diagnostics))

    def arrays(self) -> dict[str, np.ndarray]:
        a: dict[str, np.ndarray] = {
            "t": np.asarray(self.t, dtype=float),
            "truth": _stack(self.truth, 3),
            "eligible": _stack(self.eligible, 2),
            "applied": _stack(self.applied, 2),
            "sensor_t": np.asarray(self.sens_t, dtype=float),
            "sensor_pose": _stack(self.sens_pose, 3),
            "sensor_dropped": np.asarray(self.sens_dropped, dtype=bool),
        }
        c = self.c
        a["ctrl_t"] = np.asarray(c["ctrl_t"], dtype=float)
        for key, dim in (
            ("obs_pose", 3),
            ("ref_pose", 3),
            ("ref_control", 2),
            ("raw", 2),
            ("command", 2),
        ):
            a[key] = _stack(c[key], dim)
        a["obs_t_sample"] = np.asarray(c["obs_t_sample"], dtype=float)
        a["ref_s"] = np.asarray(c["ref_s"], dtype=float)
        a["solve_time"] = np.asarray(c["solve_time"], dtype=float)
        a["status"] = np.asarray(c["status"], dtype="U16")
        for key, dim in (("pred_states", 3), ("pred_controls", 2)):
            padded = _pad(c[key], dim)
            if padded is not None:
                a[key] = padded
        keys = sorted({k for d in c["diag"] for k in d})
        for k in keys:
            a[f"diag_{k}"] = np.asarray([float(d.get(k, np.nan)) for d in c["diag"]])
        return a


def _stack(rows: list[np.ndarray], dim: int) -> np.ndarray:
    return np.asarray(rows, dtype=float).reshape(-1, dim)


def _pad(rows: list[np.ndarray | None], dim: int) -> np.ndarray | None:
    if not any(r is not None for r in rows):
        return None
    n = max(len(r) for r in rows if r is not None)
    out = np.full((len(rows), n, dim), np.nan)
    for i, r in enumerate(rows):
        if r is not None:
            r = np.asarray(r, dtype=float).reshape(-1, dim)
            out[i, : len(r)] = r
    return out


# ------------------------------------------------------------------ manifest
@lru_cache(maxsize=1)
def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


def _manifest(
    sc: Scenario,
    controller: Controller,
    seed: int,
    req: RefRequest,
    label: str | None,
    termination: str,
    error: str,
    duration: float,
) -> dict[str, Any]:
    scenario = sc.model_dump(mode="json")
    ctrl = controller.describe()
    payload = json.dumps({"scenario": scenario, "controller": ctrl, "seed": seed}, sort_keys=True)
    return {
        "label": label or ctrl["name"],
        "scenario": scenario,
        "controller": ctrl,
        "seed": seed,
        "reference_request": {"mode": req.mode, "horizon": req.horizon, "dt": req.dt, "ds": req.ds},
        "duration": duration,
        "termination": termination,
        "error": error,
        "config_hash": hashlib.sha256(payload.encode()).hexdigest()[:16],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "git_commit": _git_commit(),
    }
