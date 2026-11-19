import numpy as np
import pytest

from diffdrive.controllers import make_controller
from diffdrive.evaluation import compute_metrics
from diffdrive.sim import Episode, find_episodes, run_episode


def test_save_load_roundtrip_and_metrics_recompute(tmp_path, quick_circle):
    sc = quick_circle.with_overrides({"sensor.noise_std": [0.01, 0.01, 0.005]})
    ep = run_episode(sc, make_controller("pid"), seed=1)
    ep.metrics = compute_metrics(ep)
    ep.save(tmp_path / "ep")
    loaded = Episode.load(tmp_path / "ep")
    for k, v in ep.arrays.items():
        np.testing.assert_array_equal(v, loaded.arrays[k], err_msg=k)
    again = compute_metrics(loaded)
    for k, v in ep.metrics.items():
        if isinstance(v, float):
            assert again[k] == pytest.approx(v, nan_ok=True), k
        else:
            assert again[k] == v, k
    assert find_episodes(tmp_path) == [tmp_path / "ep"]
    assert loaded.meta["config_hash"] == ep.meta["config_hash"]


def test_metrics_for_ideal_run(quick_circle):
    m = compute_metrics(run_episode(quick_circle, make_controller("feedforward")))
    assert m["success"]
    assert m["pos_rmse"] < 1e-3 and m["cte_rmse"] < 1e-3
    assert m["completion"] == pytest.approx(1.0, abs=1e-3)
    assert m["constraint_violations"] == 0
    assert m["n_failed"] == 0


def test_path_metrics_ignore_timing(quick_circle):
    """A robot that follows the path but lags behind schedule has small CTE, large time error."""
    sc = quick_circle.with_overrides({"command.latency": 0.0})
    from dataclasses import dataclass

    from diffdrive.controllers import Controller
    from diffdrive.core import Control, ControlResult, RefRequest

    class SlowFollower(Controller):
        @dataclass(frozen=True)
        class Config:
            pass

        def reference_request(self):
            return RefRequest(mode="path", horizon=0)

        def compute(self, obs, ref):
            v, _w = ref.controls[0]
            v = max(v, 0.05) * 0.5
            return ControlResult(Control(v, v * ref.curvature[0]))

    m = compute_metrics(run_episode(sc, SlowFollower()))
    assert m["cte_rmse"] < 0.02
    assert m["pos_rmse"] > 0.2
