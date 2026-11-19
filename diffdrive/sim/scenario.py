"""Validated scenario configuration (YAML ⇄ pydantic).

A scenario fixes *everything except the controller*: the map (geometry), speed
profile, initial error, plant, sensor, command pipeline, timing and failure
policy. Controllers are
compared on identical scenarios and paired seeds.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path as FsPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from diffdrive.core.limits import Limits
from diffdrive.core.paths import SCENARIOS_DIR
from diffdrive.references import SpeedProfile, Trajectory, load_map

_TICK_TOL = 1e-9


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpeedSpec(_Strict):
    cruise: float = Field(0.5, gt=0)
    accel: float = Field(0.5, gt=0)
    decel: float | None = Field(None, gt=0)
    start_speed: float = Field(0.0, ge=0)
    end_speed: float = Field(0.0, ge=0)
    lateral_accel: float | None = Field(None, gt=0)
    omega_fraction: float = Field(0.8, gt=0, le=1)
    stops: tuple[float, ...] = ()
    stop_duration: float = Field(0.0, ge=0)

    def to_profile(self) -> SpeedProfile:
        return SpeedProfile(**self.model_dump())


class TimingSpec(_Strict):
    """Clock rates. All periods must be integer multiples of ``dt_physics``."""

    dt_physics: float = Field(0.005, gt=0)
    dt_control: float = Field(0.05, gt=0)
    duration: float | None = Field(None, gt=0)  # None → reference duration + tail
    tail: float = Field(2.0, ge=0)


class InitialErrorSpec(_Strict):
    """Initial pose offset, in the body frame of the reference start pose."""

    dx: float = 0.0
    dy: float = 0.0
    dtheta: float = 0.0


class PlantSpec(_Strict):
    integrator: Literal["exact", "rk4", "euler"] = "exact"
    tau_v: float = Field(0.0, ge=0)  # first-order actuator lag [s]; 0 = none
    tau_omega: float = Field(0.0, ge=0)
    slip: float = Field(0.0, ge=0, lt=1)  # effective v = (1 - slip)·v
    omega_bias: float = 0.0  # additive yaw-rate bias [rad/s]
    process_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)  # per √s, world frame


class SensorSpec(_Strict):
    period: float | None = Field(None, gt=0)  # None → dt_control
    noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dropout: float = Field(0.0, ge=0, lt=1)
    latency: float = Field(0.0, ge=0)


class CommandSpec(_Strict):
    v_min: float = 0.0
    v_max: float = Field(1.0, gt=0)
    omega_max: float = Field(1.5, gt=0)
    accel_max: float | None = Field(None, gt=0)
    alpha_max: float | None = Field(None, gt=0)
    latency: float = Field(0.0, ge=0)

    def limits(self) -> Limits:
        return Limits(self.v_min, self.v_max, self.omega_max, self.accel_max, self.alpha_max)


class TerminationSpec(_Strict):
    """Pre-declared failure policy (doc §4.4)."""

    max_path_deviation: float = Field(1.0, gt=0)  # [m] distance to the path
    deviation_grace: float = Field(0.5, ge=0)  # [s] must persist this long


class RobotSpec(_Strict):
    """Geometry, used for drawing and for future wheel-level models."""

    length: float = 0.22
    width: float = 0.18
    wheel_radius: float = 0.033
    track_width: float = 0.16


class Scenario(_Strict):
    name: str = "unnamed"
    description: str = ""
    tags: tuple[str, ...] = ()
    map: str = "map00"  # map name in maps/ or a path to a map .yaml / .csv
    speed: SpeedSpec = SpeedSpec()
    timing: TimingSpec = TimingSpec()
    initial_error: InitialErrorSpec = InitialErrorSpec()
    plant: PlantSpec = PlantSpec()
    sensor: SensorSpec = SensorSpec()
    command: CommandSpec = CommandSpec()
    termination: TerminationSpec = TerminationSpec()
    robot: RobotSpec = RobotSpec()

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not v:
            raise ValueError("scenario name must be non-empty")
        return v

    @model_validator(mode="after")
    def _check_timing(self) -> Scenario:
        t = self.timing
        ticks(t.dt_control, t.dt_physics, "timing.dt_control")
        ticks(self.sensor_period, t.dt_physics, "sensor.period")
        ticks(self.sensor.latency, t.dt_physics, "sensor.latency", allow_zero=True)
        ticks(self.command.latency, t.dt_physics, "command.latency", allow_zero=True)
        if self.command.v_min > self.command.v_max:
            raise ValueError("command.v_min must be <= command.v_max")
        return self

    # ------------------------------------------------------------- helpers
    @property
    def sensor_period(self) -> float:
        return self.sensor.period if self.sensor.period is not None else self.timing.dt_control

    def limits(self) -> Limits:
        return self.command.limits()

    def build_reference(self) -> Trajectory:
        return Trajectory.from_path(load_map(self.map), self.speed.to_profile(), self.limits())

    def duration(self, traj: Trajectory) -> float:
        if self.timing.duration is not None:
            return self.timing.duration
        return traj.duration + self.timing.tail

    def with_overrides(self, overrides: dict[str, Any]) -> Scenario:
        return Scenario.model_validate(apply_overrides(self.model_dump(), overrides))

    # ---------------------------------------------------------------- I/O
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        data = copy.deepcopy(data)
        base = data.pop("base", None)
        if base is not None:
            merged = load_scenario(base).model_dump()
            overrides = data.pop("overrides", {})
            merged = deep_merge(merged, data)
            data = apply_overrides(merged, overrides)
        else:
            data = apply_overrides(data, data.pop("overrides", {}))
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | FsPath) -> Scenario:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        data.setdefault("name", FsPath(path).stem)
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


def ticks(period: float, dt: float, what: str, allow_zero: bool = False) -> int:
    """Convert a period to an integer number of physics ticks; fail loudly otherwise."""
    if period == 0 and allow_zero:
        return 0
    n = period / dt
    r = round(n)
    if r < 1 or not math.isclose(n, r, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"{what}={period} must be a positive integer multiple of dt_physics={dt}")
    return int(r)


def deep_merge(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-key overrides, e.g. ``{"sensor.latency": 0.05}``."""
    out = copy.deepcopy(data)
    for key, val in overrides.items():
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise KeyError(f"override {key!r}: {p!r} is not a mapping")
        node[parts[-1]] = val
    return out


def builtin_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml"))


def load_scenario(name_or_path: str | FsPath) -> Scenario:
    """Load a scenario by name (e.g. ``noisy``) or by YAML file path."""
    p = FsPath(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.is_file():
        return Scenario.from_yaml(p)
    res = SCENARIOS_DIR / f"{name_or_path}.yaml"
    if not res.is_file():
        raise FileNotFoundError(
            f"no scenario file or scenario named {name_or_path!r}; available: {builtin_scenarios()}"
        )
    return Scenario.from_yaml(res)
