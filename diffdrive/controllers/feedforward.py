"""Open-loop feed-forward: apply the reference control ``u_r``.

Not a real tracker — it has no feedback — but it is the M0 smoke test: on an
ideal plant with zero initial error it reproduces the reference (up to the
integrator), and it shows how fast disturbances accumulate without feedback.

Because the command is held for a whole control period (zero-order hold), it
uses the *average* reference control over ``[t, t + dt]`` rather than ``u_r(t)``;
holding ``u_r(t)`` would lag by ``a·dt/2`` whenever the reference accelerates.
"""

from __future__ import annotations

from diffdrive.core.types import Control, ControlResult, Observation, ReferenceWindow, RefRequest
from diffdrive.controllers.base import Controller, register


@register("feedforward")
class Feedforward(Controller):
    def reference_request(self) -> RefRequest:
        return RefRequest(mode="time", horizon=1)

    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult:
        u = 0.5 * (ref.controls[0] + ref.controls[1])
        return ControlResult(Control.from_array(u))
