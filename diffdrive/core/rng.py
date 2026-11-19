"""Independent, named, seeded random streams.

Each source of randomness (sensor noise, dropout, plant disturbance, controller
exploration, ...) gets its own generator derived from ``(seed, name)``. Adding a
new noise source, or drawing more samples from one stream, never changes the
draws of another stream — so two controllers run on the same scenario+seed see
identical exogenous noise, and enabling dropout does not change the noise values.
"""

from __future__ import annotations

import zlib

import numpy as np


class RngStreams:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._streams: dict[str, np.random.Generator] = {}

    def get(self, name: str) -> np.random.Generator:
        if name not in self._streams:
            key = zlib.crc32(name.encode("utf-8"))
            ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(key,))
            self._streams[name] = np.random.default_rng(ss)
        return self._streams[name]

    def child_seed(self, name: str) -> int:
        """A deterministic integer seed for components that want their own RNG."""
        return int(self.get(f"seed:{name}").integers(0, 2**31 - 1))
