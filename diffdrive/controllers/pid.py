"""PID tracking controller (doc §2.1) — baseline.

Control law, using the body-frame error ``[e_x, e_y, e_θ]`` of the reference
sample at window index 0 (see :func:`core.body_frame_error`)::

    v = v_r cos e_θ + PID_x(e_x)                    # longitudinal
    δ = atan(k_y · e_y / max(|v_r|, v_ε))           # lateral → heading correction
    ω = ω_r + PID_θ(wrap(e_θ + δ))                  # heading

* ``mode="time"``: index 0 is the reference at the current time, so ``e_x`` is
  the along-track lag and the longitudinal PID makes the robot keep schedule.
* ``mode="path"``: index 0 is the closest point (progress), so ``e_x ≈ 0`` and
  the speed follows the profile at the current progress (path following).

Includes integrator clamping + conditional integration (anti-windup) and a
first-order low-pass filter on the derivative term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from diffdrive.core.geometry import body_frame_error, wrap_angle
from diffdrive.core.types import Control, ControlResult, Observation, ReferenceWindow, RefRequest
from diffdrive.controllers.base import Controller, ControllerContext, register


class PIDTerm:
    """Scalar PID with integral clamp, conditional integration and filtered derivative."""

    def __init__(self, kp: float, ki: float, kd: float, i_limit: float, d_tau: float) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self.d_tau = d_tau
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error: float | None = None
        self.d_filtered = 0.0
        self._pending_integral = 0.0

    def __call__(self, e: float, dt: float) -> float:
        if self.prev_error is not None and dt > 0:
            raw_d = (e - self.prev_error) / dt
            a = dt / (self.d_tau + dt) if self.d_tau > 0 else 1.0
            self.d_filtered += a * (raw_d - self.d_filtered)
        self.prev_error = e
        self._pending_integral = max(-self.i_limit, min(self.i_limit, self.integral + e * dt))
        return self.kp * e + self.ki * self._pending_integral + self.kd * self.d_filtered

    def commit(self, accept: bool) -> None:
        """Accept the integral update unless the output saturated in the error's direction."""
        if accept:
            self.integral = self._pending_integral


@register("pid")
class PID(Controller):
    @dataclass(frozen=True)
    class Config:
        mode: Literal["time", "path"] = "time"
        # longitudinal (along-track) PID on e_x
        kp_x: float = 1.0
        ki_x: float = 0.1
        kd_x: float = 0.0
        # lateral-to-heading gain and heading PID
        k_y: float = 2.0
        kp_theta: float = 3.0
        ki_theta: float = 0.0
        kd_theta: float = 0.1
        feedforward: bool = True
        v_eps: float = 0.1  # [m/s] floor for the lateral term's speed normalisation
        creep_speed: float = 0.05  # [m/s] path mode: minimum speed until the end
        integral_limit: float = 1.0
        derivative_tau: float = 0.05  # [s] derivative low-pass time constant

    def reset(self, ctx: ControllerContext) -> None:
        super().reset(ctx)
        c = self.config
        self.pid_x = PIDTerm(c.kp_x, c.ki_x, c.kd_x, c.integral_limit, c.derivative_tau)
        self.pid_th = PIDTerm(
            c.kp_theta, c.ki_theta, c.kd_theta, c.integral_limit, c.derivative_tau
        )

    def reference_request(self) -> RefRequest:
        return RefRequest(mode=self.config.mode, horizon=0)

    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult:
        c = self.config
        dt = self.ctx.dt
        e_x, e_y, e_th = body_frame_error(obs.pose.as_array(), ref.poses[0])
        v_r, w_r = ref.controls[0]
        if c.mode == "path":
            e_x = 0.0  # progress is free in path following
            v_r = 0.0 if ref.finished else max(v_r, c.creep_speed)

        v_ff = v_r * math.cos(e_th) if c.feedforward else 0.0
        delta = math.atan(c.k_y * e_y / max(abs(v_r), c.v_eps))
        e_head = wrap_angle(e_th + delta)

        v = v_ff + self.pid_x(e_x, dt)
        w = (w_r if c.feedforward else 0.0) + self.pid_th(e_head, dt)
        if c.mode == "path" and ref.finished:
            v, w = 0.0, 0.0

        lim = self.ctx.limits
        sat = lim.saturate(Control(v, w))
        self.pid_x.commit(not (sat.v != v and e_x * (v - sat.v) > 0))
        self.pid_th.commit(not (sat.omega != w and e_head * (w - sat.omega) > 0))
        return ControlResult(
            sat,
            diagnostics={"e_x": e_x, "e_y": e_y, "e_theta": e_th, "delta": delta},
        )
