"""A complete custom controller in ~40 lines: Kanayama's kinematic tracking law.

    v = v_r cos e_θ + k_x e_x
    ω = ω_r + v_r (k_y e_y + k_θ sin e_θ)

(Y. Kanayama et al., "A stable tracking control method for an autonomous
mobile robot", ICRA 1990.)

Run it from the repository root:

    python main.py --map map02 --controller examples/my_controller.py:Kanayama --save
    python main.py --scenario lateral_offset --controller examples/my_controller.py:Kanayama \
        --param k_y=8

or add it to a benchmark suite::

    controllers:
      - controller: examples/my_controller.py:Kanayama
        label: kanayama
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from diffdrive.controllers import Controller
from diffdrive.core import Control, ControlResult, Observation, ReferenceWindow, RefRequest
from diffdrive.core.geometry import body_frame_error


class Kanayama(Controller):
    name = "kanayama"

    @dataclass(frozen=True)
    class Config:
        k_x: float = 1.0
        k_y: float = 6.0
        k_theta: float = 3.0

    def reference_request(self) -> RefRequest:
        # Time-indexed tracking, no preview needed (index 0 = reference "now").
        return RefRequest(mode="time", horizon=0)

    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult:
        c = self.config
        e_x, e_y, e_th = body_frame_error(obs.pose.as_array(), ref.poses[0])
        v_r, w_r = ref.controls[0]
        v = v_r * math.cos(e_th) + c.k_x * e_x
        w = w_r + v_r * (c.k_y * e_y + c.k_theta * math.sin(e_th))
        return ControlResult(
            self.ctx.limits.saturate(Control(v, w)),
            diagnostics={"e_x": e_x, "e_y": e_y, "e_theta": e_th},
        )
