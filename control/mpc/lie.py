import numpy as np

from control.base import Control

"""
SE(2) helpers for `LieMPC`.

A planar pose `(x, y, theta)` is an element of the Lie group SE(2): a
rotation plus a translation. The reason this matters for control: the
"naive" error between a pose and a target -- `(x - tx, y - ty)` and
`wrap_angle(theta - t_theta)` computed independently -- treats position
and heading as unrelated Euclidean quantities. The Lie-group error instead
rotates the position error into the target's own frame and derives it from
the same operation as the heading error, so both fall out of one
consistent, coordinate-invariant computation: `log(T_target^-1 @ T_pose)`.

For SE(2) specifically this "log map" is almost trivial compared to
SO(3)/SE(3) -- there's no Rodrigues-formula-style subtlety, the wrapped
rotation angle and the translation *are* the log, once the translation
has been expressed in the right frame. That's what makes SE(2) a good
first Lie group to learn on.
"""


def to_se2(pose: tuple[float, float, float]) -> np.ndarray:
    """ Build the SE(2) homogeneous matrix for a `(x, y, theta)` pose.

    @return: 3x3 matrix `[[R, t], [0, 1]]`
    """
    ## TODO: same rotation convention as `Robot._update_polygon`:
    ## [[cos(theta), -sin(theta), x],
    ##  [sin(theta),  cos(theta), y],
    ##  [0,           0,          1]]
    ...


def se2_inverse(T: np.ndarray) -> np.ndarray:
    """ Inverse of an SE(2) homogeneous matrix.

    @param T: `[[R, t], [0, 1]]`
    @return: `[[R.T, -R.T @ t], [0, 1]]`
    """
    ## TODO: use the closed form above -- don't call np.linalg.inv, it's
    ## needlessly expensive (and less numerically clean) for this shape.
    ...


def log_map(T: np.ndarray) -> np.ndarray:
    """ SE(2) logarithm: map a homogeneous matrix to a `(x, y, theta)`
    twist-like vector.

    @param T: `[[R, t], [0, 1]]`
    @return: `np.array([t_x, t_y, theta])`
    """
    ## TODO: theta = atan2(T[1, 0], T[0, 0]), wrapped via
    ## `Control._wrap_angle` (reuse it -- don't reimplement wrapping here).
    ## Translation is just `T[0, 2], T[1, 2]`.
    ...


def pose_error(
    pose: tuple[float, float, float],
    target: tuple[float, float, float],
) -> np.ndarray:
    """ Error of `pose` relative to `target`, expressed in `target`'s frame.

    This is the function `LieMPC` actually calls -- `to_se2`/`se2_inverse`/
    `log_map` above are just its building blocks.

    @param pose: `(x, y, theta)` to evaluate
    @param target: `(x, y, theta)` reference to measure error against
    @return: `np.array([e_x, e_y, e_theta])`
    """
    ## TODO: T_target = to_se2(target); T_pose = to_se2(pose)
    ## TODO: return log_map(se2_inverse(T_target) @ T_pose)
    ...
