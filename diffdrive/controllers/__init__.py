"""Controllers. Importing this package registers all built-in controllers."""

from diffdrive.controllers import feedforward, ilqr, lie, linear_mpc, lqr, nonlinear_mpc, pid, pure_pursuit
from diffdrive.controllers.base import (
    Controller,
    ControllerContext,
    available_controllers,
    get_controller_class,
    make_controller,
    register,
)

__all__ = [
    "Controller",
    "ControllerContext",
    "available_controllers",
    "feedforward",
    "get_controller_class",
    "ilqr",
    "lie",
    "linear_mpc",
    "lqr",
    "make_controller",
    "nonlinear_mpc",
    "pid",
    "pure_pursuit",
    "register",
]
