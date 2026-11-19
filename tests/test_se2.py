"""M6 acceptance tests for :mod:`core.se2` (skipped until implemented)."""

import math

import numpy as np
import pytest

from diffdrive.core import se2


@pytest.fixture(autouse=True)
def _require_impl():
    try:
        se2.exp(np.zeros(3))
    except NotImplementedError:
        pytest.skip("diffdrive.core.se2 not implemented yet (M6)")


def _poses(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([-3, -3, -math.pi + 1e-3], [3, 3, math.pi - 1e-3], size=(n, 3))


def test_hat_vee_roundtrip():
    for xi in _poses():
        assert se2.vee(se2.hat(xi)) == pytest.approx(xi)


def test_pose_matrix_roundtrip_and_inverse():
    for p in _poses():
        T = se2.from_pose(p)
        assert se2.to_pose(T) == pytest.approx(p)
        assert se2.inverse(T) @ T == pytest.approx(np.eye(3), abs=1e-12)


def test_exp_log_roundtrip_including_small_angles():
    xis = np.vstack([_poses(), [[0.3, -0.2, 1e-9], [1.0, 0.0, 0.0], [0.0, 0.0, 3.1]]])
    for xi in xis:
        assert se2.log(se2.exp(xi)) == pytest.approx(xi, abs=1e-9)


def test_exp_matches_constant_twist_motion():
    from diffdrive.models import Unicycle

    v, w, dt = 0.6, 0.9, 0.7
    T = se2.exp(np.array([v * dt, 0.0, w * dt]))
    assert se2.to_pose(T) == pytest.approx(
        Unicycle("exact").step(np.zeros(3), np.array([v, w]), dt)
    )


def test_adjoint_identity():
    """Ad(T) ξ = vee(T ξ̂ T⁻¹)."""
    for p, xi in zip(_poses(seed=1), _poses(seed=2), strict=False):
        T = se2.from_pose(p)
        assert se2.adjoint(T) @ xi == pytest.approx(se2.vee(T @ se2.hat(xi) @ se2.inverse(T)))
