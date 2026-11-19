"""The controller interface every algorithm implements — classical or learned.

Minimal example::

    from dataclasses import dataclass
    from diffdrive.controllers import Controller, register
    from diffdrive.core import Control, ControlResult

    @register("my_ctrl")
    class MyController(Controller):
        @dataclass(frozen=True)
        class Config:
            gain: float = 1.0

        def compute(self, obs, ref):
            return ControlResult(Control(ref.control.v, self.config.gain * 0.0))

Then ``python main.py --map map01 --controller my_ctrl`` (or
``path/to/file.py:MyController`` without registering) runs it through the same simulator, logger and metrics.
"""

from __future__ import annotations

import dataclasses
import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import numpy as np

from diffdrive.core.limits import Limits
from diffdrive.core.types import ControlResult, Observation, ReferenceWindow, RefRequest
from diffdrive.models.unicycle import Unicycle


@dataclass(frozen=True)
class ControllerContext:
    """Episode information given to a controller at :meth:`Controller.reset`.

    Attributes:
        dt: control period [s]; ``compute`` is called every ``dt`` seconds.
        limits: command limits (also enforced downstream by the simulator).
        model: the *nominal* prediction model. It may differ from the true plant.
        rng: a controller-private random generator (for stochastic policies).
        seed: the integer seed used to create ``rng``.
    """

    dt: float
    limits: Limits
    model: Unicycle
    rng: np.random.Generator
    seed: int


class Controller(ABC):
    """Base class for all controllers.

    Lifecycle per episode: ``reset(ctx)`` → ``reference_request()`` → repeated
    ``compute(obs, ref)``. Subclasses typically override:

    * ``Config`` — a (frozen) dataclass of tunable parameters;
    * ``reference_request`` — how much preview they need, time- or path-indexed;
    * ``reset`` — call ``super().reset(ctx)`` then clear internal state;
    * ``compute`` — return a :class:`ControlResult`.

    A controller must not mutate simulator state; it only sees the observation
    (a measurement, never ground truth) and the reference window.
    """

    name: ClassVar[str] = ""
    implemented: ClassVar[bool] = True
    milestone: ClassVar[str] = ""

    @dataclass(frozen=True)
    class Config:
        pass

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        self.config = build_config(type(self).Config, config, overrides)
        self._ctx: ControllerContext | None = None

    @property
    def ctx(self) -> ControllerContext:
        if self._ctx is None:
            raise RuntimeError(f"{type(self).__name__}.reset(ctx) must be called before use")
        return self._ctx

    def reset(self, ctx: ControllerContext) -> None:
        self._ctx = ctx

    def reference_request(self) -> RefRequest:
        return RefRequest(mode="time", horizon=0)

    @abstractmethod
    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult: ...

    def describe(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "name": self.name or type(self).__name__,
            "class": f"{type(self).__module__}:{type(self).__qualname__}",
            "config": _jsonable(dataclasses.asdict(cfg)) if dataclasses.is_dataclass(cfg) else {},
        }


C = TypeVar("C")


def build_config(cls: type[C], config: Any, overrides: Mapping[str, Any]) -> C:
    if config is None:
        data: dict[str, Any] = {}
    elif isinstance(config, cls):
        data = dataclasses.asdict(config)  # type: ignore[call-overload]
    elif isinstance(config, Mapping):
        data = dict(config)
    else:
        raise TypeError(f"config must be {cls.__name__}, a mapping or None; got {type(config)}")
    data.update(overrides)
    fields = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - fields
    if unknown:
        raise TypeError(
            f"unknown config key(s) {sorted(unknown)} for {cls.__qualname__}; "
            f"valid keys: {sorted(fields)}"
        )
    return cls(**data)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


# ------------------------------------------------------------------- registry
_REGISTRY: dict[str, type[Controller]] = {}

T = TypeVar("T", bound=type[Controller])


def register(name: str) -> Callable[[T], T]:
    """Class decorator registering a controller under ``name``."""

    def deco(cls: T) -> T:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"controller name {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def available_controllers() -> dict[str, type[Controller]]:
    return dict(sorted(_REGISTRY.items()))


def _load_module_from_file(path: str):
    import importlib.util
    import sys
    from pathlib import Path

    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(path)
    name = f"_user_controller_{p.stem}_{abs(hash(str(p)))}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def get_controller_class(spec: str) -> type[Controller]:
    """Resolve a controller class from

    * a registry name (``pid``),
    * an import path (``my_package.controllers:MyController``), or
    * a Python file (``path/to/my_controller.py:MyController``).
    """
    if ":" in spec:
        mod, _, attr = spec.rpartition(":")
        module = (
            _load_module_from_file(mod) if mod.endswith(".py") else importlib.import_module(mod)
        )
        cls = getattr(module, attr)
        if not (isinstance(cls, type) and issubclass(cls, Controller)):
            raise TypeError(f"{spec} is not a Controller subclass")
        return cls
    ctrls = available_controllers()
    if spec not in ctrls:
        raise KeyError(f"unknown controller {spec!r}; available: {sorted(ctrls)}")
    return ctrls[spec]


def make_controller(spec: str, config: Mapping[str, Any] | None = None) -> Controller:
    return get_controller_class(spec)(config)
