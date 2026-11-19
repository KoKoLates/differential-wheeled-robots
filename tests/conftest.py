import matplotlib

matplotlib.use("Agg")

import pytest
from helpers import circle, line, map_file

import diffdrive.controllers  # noqa: F401  (registers builtin controllers)
from diffdrive.sim import Scenario


@pytest.fixture(scope="session")
def maps_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("maps")


@pytest.fixture(scope="session")
def short_line(maps_dir) -> str:
    """A 1.5 m line map (path to its .yaml)."""
    return map_file(maps_dir, "short_line", line(1.5))


@pytest.fixture(scope="session")
def line_map(maps_dir) -> str:
    return map_file(maps_dir, "line", line(6.0))


@pytest.fixture
def quick_circle(maps_dir) -> Scenario:
    """Half a lap of a 1 m circle, nominal conditions, for fast closed-loop tests."""
    m = map_file(maps_dir, "half_circle", circle(1.0, laps=0.5))
    return Scenario.from_dict({"name": "quick_circle", "map": m})
