"""Benchmark suites: scenarios × controllers × seeds, with parameter sweeps.

Example ``suite.yaml``::

    name: latency_sweep
    seeds: [0, 1, 2]
    maps: [map01, map02]          # optional: run every scenario on these maps
    scenarios:
      - nominal
      - scenario: delayed
        label: sensor_latency
        sweep: {sensor.latency: [0.0, 0.05, 0.1]}
      - scenario: lateral_offset
        maps: [map01]             # per-scenario maps override the suite's
    controllers:
      - pid
      - controller: pure_pursuit
        config: {lookahead: 0.4}

Without ``maps`` a scenario runs on its own default map. Every controller sees
the same maps, scenario variants and seeds (paired comparison). Results land in ``<out>/results.csv`` plus one episode directory
per run.
"""

from __future__ import annotations

import csv
import itertools
import math
import re
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from diffdrive.core.paths import SUITES_DIR
from diffdrive.sim.scenario import Scenario, load_scenario


class ControllerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    controller: str
    label: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    nominal_integrator: str | None = None

    @property
    def name(self) -> str:
        return self.label or self.controller.split(":")[-1]


class ScenarioEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str
    label: str | None = None
    maps: list[str] | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    sweep: dict[str, list[Any]] = Field(default_factory=dict)


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    seeds: list[int] = Field(default_factory=lambda: [0])
    maps: list[str] | None = None
    scenarios: list[ScenarioEntry]
    controllers: list[ControllerEntry]

    @field_validator("scenarios", mode="before")
    @classmethod
    def _scen(cls, v: list[Any]) -> list[Any]:
        return [{"scenario": x} if isinstance(x, str) else x for x in v]

    @field_validator("controllers", mode="before")
    @classmethod
    def _ctrl(cls, v: list[Any]) -> list[Any]:
        return [{"controller": x} if isinstance(x, str) else x for x in v]

    @classmethod
    def from_yaml(cls, path: str | FsPath) -> Suite:
        p = FsPath(path)
        if not p.is_file():
            res = SUITES_DIR / f"{path}.yaml"
            if not res.is_file():
                raise FileNotFoundError(f"no suite file or builtin named {path!r}")
            return cls.model_validate(yaml.safe_load(res.read_text()))
        return cls.model_validate(yaml.safe_load(p.read_text()))

    def expand(self) -> list[RunSpec]:
        runs = []
        labels_seen: set[str] = set()
        for se in self.scenarios:
            base0 = load_scenario(se.scenario).with_overrides(se.overrides)
            label = se.label or base0.name
            keys = list(se.sweep)
            for map_name in se.maps or self.maps or [base0.map]:
                base = base0.with_overrides({"map": map_name})
                for values in itertools.product(*(se.sweep[k] for k in keys)) if keys else [()]:
                    params = dict(zip(keys, values, strict=False))
                    sc = base.with_overrides(params) if params else base
                    var_label = f"{map_name}/{label}" + (
                        "[" + ",".join(f"{k}={v}" for k, v in params.items()) + "]"
                        if params
                        else ""
                    )
                    if var_label in labels_seen:
                        raise ValueError(f"duplicate scenario label {var_label!r}; set `label:`")
                    labels_seen.add(var_label)
                    for ce in self.controllers:
                        for seed in self.seeds:
                            runs.append(RunSpec(sc, var_label, params, ce, seed))
        names = [c.name for c in self.controllers]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate controller labels {names}; set `label:`")
        return runs


@dataclass
class RunSpec:
    scenario: Scenario
    scenario_label: str
    params: dict[str, Any]
    controller: ControllerEntry
    seed: int

    def directory(self, root: FsPath) -> FsPath:
        return (
            root
            / "episodes"
            / _slug(self.scenario_label)
            / _slug(self.controller.name)
            / f"seed_{self.seed}"
        )


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=,-]+", "_", s).strip("_")


def run_one(spec: RunSpec, root: FsPath | None, save_episode: bool = True) -> dict[str, Any]:
    from diffdrive.controllers import make_controller
    from diffdrive.sim.runner import run_episode
    from diffdrive.evaluation.metrics import compute_metrics

    ctrl = make_controller(spec.controller.controller, spec.controller.config)
    t0 = time.perf_counter()
    ep = run_episode(
        spec.scenario,
        ctrl,
        seed=spec.seed,
        label=spec.controller.name,
        raise_errors=False,
        nominal_integrator=spec.controller.nominal_integrator,
    )
    ep.meta["scenario_label"] = spec.scenario_label
    ep.meta["params"] = spec.params
    ep.metrics = compute_metrics(ep)
    row: dict[str, Any] = {
        "scenario": spec.scenario_label,
        "scenario_base": spec.scenario.name,
        "map": spec.scenario.map,
        "controller": spec.controller.name,
        "seed": spec.seed,
        **{f"param:{k}": v for k, v in spec.params.items()},
        **ep.metrics,
        "error": ep.meta.get("error", ""),
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    if root is not None and save_episode:
        d = spec.directory(root)
        ep.save(d)
        row["episode"] = str(d.relative_to(root))
    return row


def _run_one_star(args: tuple[RunSpec, FsPath | None, bool]) -> dict[str, Any]:
    return run_one(*args)


def run_suite(
    suite: Suite,
    out: str | FsPath | None,
    jobs: int = 1,
    save_episodes: bool = True,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    specs = suite.expand()
    root = FsPath(out) if out is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "suite.yaml").write_text(
            yaml.safe_dump(suite.model_dump(mode="json"), sort_keys=False)
        )
    rows: list[dict[str, Any]] = []
    args = [(s, root, save_episodes) for s in specs]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, row in enumerate(pool.map(_run_one_star, args)):
                rows.append(row)
                if progress:
                    progress(i + 1, len(specs), row)
    else:
        for i, a in enumerate(args):
            row = _run_one_star(a)
            rows.append(row)
            if progress:
                progress(i + 1, len(specs), row)
    if root is not None:
        write_results(rows, root / "results.csv")
    return rows


def write_results(rows: list[dict[str, Any]], path: FsPath) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in keys})


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


def read_results(path: str | FsPath) -> list[dict[str, Any]]:
    p = FsPath(path)
    if p.is_dir():
        p = p / "results.csv"
    with open(p, newline="") as f:
        return [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _parse(v: str) -> Any:
    if v == "":
        return math.nan
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v
