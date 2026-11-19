import math

import numpy as np
import pytest

from diffdrive.core import Control, Limits, body_frame_error, compose_body_offset, wrap_angle


@pytest.mark.parametrize(
    ("a", "expected"),
    [
        (0.0, 0.0),
        (math.pi, -math.pi),
        (-math.pi, -math.pi),
        (3 * math.pi, -math.pi),
        (2.0, 2.0),
        (-4.0, -4.0 + 2 * math.pi),
    ],
)
def test_wrap_angle_scalar(a, expected):
    assert wrap_angle(a) == pytest.approx(expected)


def test_wrap_angle_array_range():
    a = np.linspace(-20, 20, 1001)
    w = wrap_angle(a)
    assert np.all(w >= -math.pi) and np.all(w < math.pi)
    assert np.allclose(np.cos(w), np.cos(a)) and np.allclose(np.sin(w), np.sin(a))


def test_body_frame_error_convention():
    pose = np.array([0.0, 0.0, math.pi / 2])  # facing +y
    ref = np.array([0.0, 1.0, math.pi / 2])  # 1 m straight ahead
    ex, ey, eth = body_frame_error(pose, ref)
    assert (
        ex == pytest.approx(1.0)
        and ey == pytest.approx(0.0, abs=1e-12)
        and eth == pytest.approx(0.0)
    )
    ref_left = np.array([-1.0, 0.0, math.pi / 2])  # 1 m to the robot's left
    assert body_frame_error(pose, ref_left)[1] == pytest.approx(1.0)


def test_body_frame_error_wraps_near_pi():
    e = body_frame_error(np.array([0, 0, math.pi - 0.01]), np.array([0, 0, -math.pi + 0.01]))
    assert e[2] == pytest.approx(0.02)


def test_compose_body_offset():
    p = compose_body_offset(np.array([1.0, 2.0, math.pi / 2]), 0.0, 0.5, 0.1)
    assert p == pytest.approx([0.5, 2.0, math.pi / 2 + 0.1])


def test_limits_saturate_and_rate():
    lim = Limits(v_min=0.0, v_max=1.0, omega_max=1.0, accel_max=1.0, alpha_max=2.0)
    u = lim.apply(Control(5.0, -5.0), Control(0.0, 0.0), dt=0.1)
    assert u.v == pytest.approx(0.1) and u.omega == pytest.approx(-0.2)
    assert lim.violation(Control(1.5, 0.0)) == pytest.approx(0.5)
    assert lim.violation(Control(0.5, 0.5)) == 0.0
    with pytest.raises(ValueError):
        Limits(v_min=1.0, v_max=0.5)
