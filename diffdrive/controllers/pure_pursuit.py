"""Pure Pursuit for a unicycle (doc §2.2) — baseline, path-following track.

The lookahead target is chosen *ahead in arclength* from the monotonic progress
estimate (never by nearest Euclidean point, which would jump across a figure-
eight crossing). With the target ``(x_L, y_L)`` in the body frame and
``L = ‖(x_L, y_L)‖``::

    κ_c = 2 y_L / L²,    ω = v κ_c,    L_d = clip(L₀ + k_v |v|, L_min, L_max)

Speed follows the reference speed profile at the current progress (with a small
creep speed so a profile that starts at rest can get moving).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from diffdrive.core.geometry import world_to_body
from diffdrive.core.types import Control, ControlResult, Observation, ReferenceWindow, RefRequest
from diffdrive.controllers.base import Controller, register


@register("pure_pursuit")
class PurePursuit(Controller):
    @dataclass(frozen=True)
    class Config:
        lookahead: float = 0.3  # L₀ [m]
        lookahead_gain: float = 0.5  # k_v [s]
        lookahead_min: float = 0.15
        lookahead_max: float = 1.5
        spacing: float = 0.02  # reference window spacing [m]
        creep_speed: float = 0.05  # [m/s]
        reduce_speed_in_turns: bool = True  # cap v so that |ω| ≤ ω_max

    def reference_request(self) -> RefRequest:
        c = self.config
        horizon = math.ceil(c.lookahead_max / c.spacing) + 1
        return RefRequest(mode="path", horizon=horizon, ds=c.spacing)

    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult:
        c = self.config
        if ref.finished:
            return ControlResult(Control(0.0, 0.0), diagnostics={"lookahead": 0.0})

        v = max(float(ref.controls[0, 0]), c.creep_speed)
        ld = float(
            np.clip(c.lookahead + c.lookahead_gain * abs(v), c.lookahead_min, c.lookahead_max)
        )
        ahead = ref.s - ref.s[0]
        j = int(np.searchsorted(ahead, ld))
        j = min(j, len(ahead) - 1)
        target = ref.poses[j, :2]

        x_l, y_l = world_to_body(obs.pose.as_array(), target)
        d2 = x_l * x_l + y_l * y_l
        kappa = 2.0 * y_l / d2 if d2 > 1e-9 else 0.0
        if c.reduce_speed_in_turns and abs(kappa) > 1e-9:
            v = min(v, self.ctx.limits.omega_max / abs(kappa))
        w = v * kappa
        lim = self.ctx.limits
        return ControlResult(
            lim.saturate(Control(v, w)),
            predicted_states=None,
            diagnostics={
                "lookahead": ld,
                "kappa": kappa,
                "target_x": target[0],
                "target_y": target[1],
            },
        )
