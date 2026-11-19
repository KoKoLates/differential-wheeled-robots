"""maps/generate.py is standalone; the CSV it writes is the contract with the simulator."""

import importlib.util

import numpy as np
import pytest

from diffdrive.core.paths import MAPS_DIR
from diffdrive.references import load_map


@pytest.fixture
def gen(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("map_generate", MAPS_DIR / "generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "MAPS_DIR", tmp_path)
    return mod


def test_shapes_and_random_maps_load_in_simulator(gen, tmp_path):
    gen.main(["--shape", "circle", "--radius", "1.0", "--no-catalogue"])
    gen.main(["--random", "--points", "5", "--seed", "3", "--min-radius", "0.5"])
    files = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("map*"))
    assert files == [
        "csv/map00.csv", "csv/map01.csv", "png/map00.png", "png/map01.png",
        "yaml/map00.yaml", "yaml/map01.yaml",
    ]  # fmt: skip
    assert "png/map01.png" in (tmp_path / "README.md").read_text()

    c = load_map(tmp_path / "yaml" / "map00.yaml")
    assert c.length == pytest.approx(2 * np.pi, rel=1e-3)
    assert np.allclose(c.kappa[5:-5], 1.0, atol=1e-2)
    r = load_map(tmp_path / "yaml" / "map01.yaml")
    assert np.abs(r.kappa).max() <= 1 / 0.5 + 1e-9


def test_random_maps_are_reproducible_and_fixtures_protected(gen, tmp_path):
    gen.main(["--random", "--seed", "7", "--name", "a", "--no-catalogue"])
    gen.main(["--random", "--seed", "7", "--name", "b", "--no-catalogue"])
    a, b = (np.loadtxt(tmp_path / "csv" / f"{n}.csv", delimiter=",", skiprows=1) for n in "ab")
    assert np.array_equal(a, b)
    with pytest.raises(SystemExit, match="exists"):
        gen.main(["--random", "--seed", "8", "--name", "a", "--no-catalogue"])
