"""Command limits shared by the actuator safety layer and the controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffdrive.core.types import Control


@dataclass(frozen=True)
class Limits:
    """Velocity-command bounds and optional slew-rate (acceleration) bounds.

    Controllers receive these so they can respect them natively (e.g. as MPC
    constraints). The simulator *also* enforces them in the actuator layer and
    logs both the raw and the limited command, so a controller that ignores them
    is visible in the constraint-violation metrics.
    """

    v_min: float = 0.0
    v_max: float = 1.0
    omega_max: float = 1.5
    accel_max: float | None = None  # |Δv| ≤ accel_max·dt
    alpha_max: float | None = None  # |Δω| ≤ alpha_max·dt

    def __post_init__(self) -> None:
        if not self.v_min <= self.v_max:
            raise ValueError(f"v_min ({self.v_min}) must be <= v_max ({self.v_max})")
        if self.omega_max <= 0:
            raise ValueError("omega_max must be positive")
        for name in ("accel_max", "alpha_max"):
            val = getattr(self, name)
            if val is not None and val <= 0:
                raise ValueError(f"{name} must be positive or None")

    def saturate(self, u: Control) -> Control:
        return Control(
            float(np.clip(u.v, self.v_min, self.v_max)),
            float(np.clip(u.omega, -self.omega_max, self.omega_max)),
        )

    def apply(self, u: Control, previous: Control, dt: float) -> Control:
        """Saturate, then rate-limit relative to ``previous`` (the last limited command)."""
        u = self.saturate(u)
        v, w = u.v, u.omega
        if self.accel_max is not None:
            dv = self.accel_max * dt
            v = float(np.clip(v, previous.v - dv, previous.v + dv))
        if self.alpha_max is not None:
            dw = self.alpha_max * dt
            w = float(np.clip(w, previous.omega - dw, previous.omega + dw))
        return Control(v, w)

    def violation(self, u: Control) -> float:
        """Largest amount by which ``u`` exceeds the magnitude bounds (0 if feasible)."""
        return max(
            0.0,
            self.v_min - u.v,
            u.v - self.v_max,
            abs(u.omega) - self.omega_max,
        )
