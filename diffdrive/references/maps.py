"""Load reference maps produced by ``maps/generate.py`` (or any tool writing the same format).

A map is ``maps/yaml/mapNN.yaml`` (metadata; ``points`` is the CSV path relative
to the yaml) + ``maps/csv/mapNN.csv`` with columns ``s, x, y, yaw, kappa`` on an
arclength grid. The simulator never knows how a
map was generated; the CSV is the whole contract.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path as FsPath
from typing import Any

import numpy as np
import yaml

from diffdrive.core.paths import MAPS_DIR
from diffdrive.references.path import Path

COLUMNS = ("s", "x", "y", "yaw", "kappa")


def available_maps() -> list[str]:
    return sorted(p.stem for p in (MAPS_DIR / "yaml").glob("map*.yaml"))


def _resolve(name_or_file: str | FsPath) -> tuple[FsPath, dict[str, Any]]:
    """Return ``(csv_path, metadata)`` for a map name, a ``.yaml`` file or a ``.csv`` file."""
    p = FsPath(name_or_file)
    if p.suffix == ".csv":
        if not p.is_file():
            raise FileNotFoundError(p)
        return p, {}
    yml = p if p.suffix in (".yaml", ".yml") else MAPS_DIR / "yaml" / f"{name_or_file}.yaml"
    if not yml.is_file():
        raise FileNotFoundError(
            f"no map file or map named {str(name_or_file)!r}; available: {available_maps()}"
        )
    meta = yaml.safe_load(yml.read_text()) or {}
    return yml.parent / meta.get("points", f"{yml.stem}.csv"), meta


def map_info(name_or_file: str | FsPath) -> dict[str, Any]:
    return _resolve(name_or_file)[1]


def load_map(name_or_file: str | FsPath) -> Path:
    """Load a map as a :class:`Path` (heading is unwrapped so it can be interpolated)."""
    csv, _ = _resolve(name_or_file)
    return _load_csv(str(csv.resolve()), csv.stat().st_mtime_ns)


@lru_cache(maxsize=64)
def _load_csv(file: str, _mtime: int) -> Path:
    with open(file) as f:
        header = [h.strip() for h in f.readline().split(",")]
    if tuple(header) != COLUMNS:
        raise ValueError(f"{file}: expected header {','.join(COLUMNS)}, got {','.join(header)}")
    d = np.loadtxt(file, delimiter=",", skiprows=1, ndmin=2)
    return Path(d[:, 0], d[:, 1:3], np.unwrap(d[:, 3]), d[:, 4])


def save_map(path: Path, file: str | FsPath, description: str = "") -> FsPath:
    """Write ``path`` as ``<file>.csv`` + ``<file>.yaml`` (used by tests and scripts)."""
    f = FsPath(file).with_suffix("")
    f.parent.mkdir(parents=True, exist_ok=True)
    yaw = (path.theta + np.pi) % (2 * np.pi) - np.pi
    data = np.column_stack([path.s, path.xy, yaw, path.kappa])
    csv = f.with_suffix(".csv")
    np.savetxt(csv, data, delimiter=",", header=",".join(COLUMNS), comments="", fmt="%.6f")
    meta = {"description": description, "points": csv.name, "length": round(path.length, 3)}
    f.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    return f.with_suffix(".yaml")
