"""Plant, sensor and command-pipeline components used by the runner.

Signal chain for one command (doc §5A.5)::

    controller → raw command → Limits (saturation + slew) → command delay queue
               → eligible command → actuator lag → applied velocity → plant
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np

from diffdrive.core.geometry import wrap_angle
from diffdrive.core.limits import Limits
from diffdrive.core.types import Array, Control
from diffdrive.models.unicycle import Unicycle
from diffdrive.sim.scenario import PlantSpec, SensorSpec

T = TypeVar("T")


class DelayQueue(Generic[T]):
    """Items become available at an integer tick; FIFO for equal due ticks."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, T]] = []
        self._n = 0

    def push(self, due_tick: int, item: T) -> None:
        heapq.heappush(self._heap, (due_tick, self._n, item))
        self._n += 1

    def pop_due(self, tick: int) -> list[T]:
        out = []
        while self._heap and self._heap[0][0] <= tick:
            out.append(heapq.heappop(self._heap)[2])
        return out

    def __len__(self) -> int:
        return len(self._heap)


class UnicyclePlant:
    """Ground-truth robot: unicycle kinematics + optional actuator lag, slip, bias, noise.

    The plant state is the pose plus the *actual* velocities ``(v, ω)``; with
    zero lag the actual velocity equals the eligible command.
    """

    def __init__(self, spec: PlantSpec, rng: np.random.Generator) -> None:
        self.spec = spec
        self.model = Unicycle(spec.integrator)
        self.rng = rng
        self.pose = np.zeros(3)
        self.velocity = np.zeros(2)

    def reset(self, pose: Array) -> None:
        self.pose = np.asarray(pose, dtype=float).copy()
        self.velocity = np.zeros(2)

    def advance(self, eligible: Control, dt: float) -> Array:
        """Advance one physics tick; returns the applied (actuator-output) velocity."""
        cmd = eligible.as_array()
        for i, tau in enumerate((self.spec.tau_v, self.spec.tau_omega)):
            if tau > 0:
                self.velocity[i] += (cmd[i] - self.velocity[i]) * (1.0 - math.exp(-dt / tau))
            else:
                self.velocity[i] = cmd[i]
        applied = self.velocity.copy()
        effective = np.array(
            [(1.0 - self.spec.slip) * applied[0], applied[1] + self.spec.omega_bias]
        )
        self.pose = self.model.step(self.pose, effective, dt)
        std = np.asarray(self.spec.process_noise_std)
        if np.any(std > 0):
            self.pose = self.pose + std * math.sqrt(dt) * self.rng.standard_normal(3)
            self.pose[2] = wrap_angle(float(self.pose[2]))
        return applied


@dataclass(frozen=True)
class SensorSample:
    pose: Array
    t_sample: float
    tick_sample: int


@dataclass
class PoseSensor:
    """Noisy, biased, possibly dropped pose measurements delivered after a latency.

    Noise and dropout use separate RNG streams, and noise is drawn even for
    dropped samples, so changing the dropout rate never changes noise values.
    """

    spec: SensorSpec
    period_ticks: int
    latency_ticks: int
    noise_rng: np.random.Generator
    dropout_rng: np.random.Generator
    queue: DelayQueue[SensorSample] = field(default_factory=DelayQueue)

    def measure(self, truth: Array, tick: int, t: float) -> SensorSample:
        noise = np.asarray(self.spec.noise_std) * self.noise_rng.standard_normal(3)
        z = truth + np.asarray(self.spec.bias) + noise
        z[2] = wrap_angle(float(z[2]))
        return SensorSample(z, t, tick)

    def is_sample_tick(self, tick: int) -> bool:
        return tick % self.period_ticks == 0

    def sample(self, truth: Array, tick: int, t: float) -> tuple[SensorSample, bool]:
        """Measure now and queue for delivery unless dropped. Returns ``(sample, dropped)``."""
        sample = self.measure(truth, tick, t)
        dropped = bool(self.dropout_rng.random() < self.spec.dropout)
        if not dropped:
            self.queue.push(tick + self.latency_ticks, sample)
        return sample, dropped

    def deliver(self, tick: int) -> list[SensorSample]:
        return self.queue.pop_due(tick)


class CommandPipeline:
    """Shared actuator safety layer + command transport delay."""

    def __init__(self, limits: Limits, dt_control: float, latency_ticks: int) -> None:
        self.limits = limits
        self.dt_control = dt_control
        self.latency_ticks = latency_ticks
        self.queue: DelayQueue[Control] = DelayQueue()
        self.last_limited = Control.zero()
        self.eligible = Control.zero()

    def submit(self, raw: Control, tick: int) -> Control:
        limited = self.limits.apply(raw, self.last_limited, self.dt_control)
        self.last_limited = limited
        self.queue.push(tick + self.latency_ticks, limited)
        return limited

    def update(self, tick: int) -> Control:
        for cmd in self.queue.pop_due(tick):
            self.eligible = cmd
        return self.eligible
