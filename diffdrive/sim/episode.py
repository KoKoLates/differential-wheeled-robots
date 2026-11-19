"""Episode logs: the single source of truth for metrics, plots and datasets.

An episode is saved as two files in one directory::

    manifest.json   # scenario, controller config, seed, versions, termination, metrics
    episode.npz     # typed arrays (see the field list on :class:`Episode`)

Metrics and plots are computed *from these logs*, never from live simulator
state, so any result can be recomputed or re-plotted later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path as FsPath
from typing import Any

import numpy as np

from diffdrive.core.types import Array
from diffdrive.references.trajectory import Trajectory

SCHEMA_VERSION = 1


@dataclass
class Episode:
    """Arrays of one closed-loop run.

    Physics-tick arrays (length N, one row per ``dt_physics``):
        ``t``, ``truth`` (N,3), ``eligible`` (N,2) command reaching the actuator,
        ``applied`` (N,2) actuator-output velocity during ``[t, t+dt)``.

    Control-tick arrays (length K, one row per ``dt_control``):
        ``ctrl_t``, ``obs_pose`` (K,3), ``obs_t_sample``, ``ref_pose`` (K,3) and
        ``ref_control`` (K,2) at window index 0, ``ref_s``, ``raw`` (K,2) controller
        output, ``command`` (K,2) after the limiter, ``solve_time`` [s], ``status``,
        optional ``pred_states`` (K,H+1,3) / ``pred_controls`` (K,H,2) NaN-padded,
        and ``diag_<name>`` per diagnostic key.

    Sensor arrays (length M, every sensor tick): ``sensor_t``, ``sensor_pose``
    (M,3) and ``sensor_dropped``.

    ``reference`` holds the full trajectory so plots/metrics need nothing else.
    """

    meta: dict[str, Any]
    arrays: dict[str, Array]
    reference: Trajectory
    metrics: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Array:
        return self.arrays[key]

    @property
    def label(self) -> str:
        return str(self.meta.get("label") or self.meta["controller"]["name"])

    @property
    def scenario_name(self) -> str:
        return str(self.meta["scenario"]["name"])

    @property
    def map_name(self) -> str:
        return str(self.meta["scenario"].get("map", ""))

    @property
    def title(self) -> str:
        """``"<map> · <scenario>"`` for figure titles."""
        return f"{self.map_name} · {self.scenario_name}" if self.map_name else self.scenario_name

    @property
    def termination(self) -> str:
        return str(self.meta["termination"])

    # ---------------------------------------------------------------- I/O
    def save(self, directory: str | FsPath) -> FsPath:
        d = FsPath(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = dict(self.arrays)
        arrays.update({f"ref__{k}": v for k, v in self.reference.to_arrays().items()})
        np.savez_compressed(d / "episode.npz", **arrays)
        manifest = {"schema_version": SCHEMA_VERSION, **self.meta, "metrics": self.metrics}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
        return d

    @classmethod
    def load(cls, directory: str | FsPath) -> Episode:
        d = FsPath(directory)
        manifest = json.loads((d / "manifest.json").read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{d}: unsupported schema version {manifest.get('schema_version')}")
        metrics = manifest.pop("metrics", {})
        manifest.pop("schema_version")
        with np.load(d / "episode.npz", allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        ref = Trajectory.from_arrays({k[5:]: v for k, v in data.items() if k.startswith("ref__")})
        arrays = {k: v for k, v in data.items() if not k.startswith("ref__")}
        return cls(manifest, arrays, ref, metrics)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def find_episodes(root: str | FsPath) -> list[FsPath]:
    """All episode directories under ``root`` (or ``root`` itself)."""
    root = FsPath(root)
    if (root / "manifest.json").is_file():
        return [root]
    return sorted(p.parent for p in root.rglob("manifest.json"))
