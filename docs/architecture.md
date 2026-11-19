# Architecture

The project is one deterministic simulator, one controller interface, and one
evaluation pipeline shared by every algorithm — classical, optimisation-based or
learned. This document describes how the pieces fit, the conventions they rely
on, and the decisions that differ from (or refine) the
[benchmark specification](differential_drive_control_benchmark.md).

## Design rules

1. **A controller only sees an `Observation` and a `ReferenceWindow`, and only
   returns a `ControlResult`.** It never reads ground truth, never mutates the
   simulator, and never decides when to "advance a waypoint".
2. **Everything that is not the algorithm belongs to the framework**: timing,
   latency, noise, actuator limits, reference sampling, progress tracking,
   logging, metrics and plots.
3. **Logs are the source of truth.** Metrics, figures, animations (and later,
   datasets) are computed from saved episodes, so every result can be recomputed.
4. **Determinism.** Integer-tick time, named seeded RNG streams, no global RNG.
   Two controllers run with the same scenario and seed see identical noise.
5. **Fail loudly.** Invalid configs, non-integer clock ratios and infeasible
   references raise errors instead of being silently rounded or clipped.

## Layout and dependency direction

All library code lives in the `diffdrive/` package (imported as `diffdrive.<module>`);
the entry scripts sit at the repository root and are run directly
(`python main.py ...`), so nothing has to be installed. Data that you edit by hand
(maps, scenarios, suites) lives in `configs/`.

```text
core  ←  models  ←  references  ←  controllers  ←  sim  ←  evaluation  ←  viz  ←  scripts
```

| Package | Responsibility |
|---|---|
| `core` | Data types (`Pose2D`, `Control`, `Observation`, `ReferenceWindow`, `ControlResult`, `RefRequest`), geometry (`wrap_angle`, `body_frame_error`, …), `Limits`, `RngStreams`, SE(2) template |
| `models` | `Unicycle` discrete model (`exact` ZOH / `rk4` / `euler`), rollouts, finite-difference Jacobians |
| `references` | `Path` (arclength-parameterised curve), `SpeedProfile` → `Trajectory` (timed `q_r(t), u_r(t)`), `ProgressTracker`, map loading (`load_map`), feasibility checks |
| `controllers` | `Controller` base class, registry, baselines (feedforward, PID, Pure Pursuit) and templates (LQR, iLQR, LMPC, NMPC, Lie-LQR, Lie-MPC) |
| `sim` | `Scenario` config (pydantic), plant / sensor / command pipeline, `run_episode`, `Episode` log format |
| `evaluation` | metrics from logs, benchmark suites (scenario × controller × seed, sweeps), aggregation |
| `viz` | static figures, animations (with predicted horizons) |
| scripts | `main.py` (one episode), `benchmark.py` (suites), `visualize.py` (reports / comparisons / GIFs), `export_dataset.py` (learning data) |
| `maps/` | `yaml/mapNN.yaml` + `csv/mapNN.csv` (`s,x,y,yaw,kappa`) + `png/` previews, standalone `generate.py` |
| `configs/` | `scenarios/` (conditions + default map), `suites/` (YAML) |

`learning/` (dataset export, Gymnasium adapter, BC / DAgger) will sit on top of
`sim` + `evaluation` without changing them — see the [roadmap](ROADMAP.md).

## Conventions

* SI units (m, s, rad). World frame x-forward / y-left, heading CCW from +x.
* Pose `[x, y, θ]`, control `[v, ω]`. Headings are wrapped to `[-π, π)`;
  *differences* are always wrapped. Paths store unwrapped heading internally so
  they can be interpolated.
* Body-frame tracking error (spec §1.4):
  `[e_x, e_y] = R(θ)ᵀ (p_r − p)`, `e_θ = wrap(θ_r − θ)`. `e_x > 0` means the
  reference is ahead, `e_y > 0` means it is to the left. Implemented once in
  `core.geometry.body_frame_error`.

## The controller interface

```python
class Controller(ABC):
    name: ClassVar[str]               # set by @register("name")
    implemented: ClassVar[bool] = True
    @dataclass(frozen=True)
    class Config: ...                 # tunables; unknown keys are rejected

    def reset(self, ctx: ControllerContext) -> None      # dt, limits, nominal model, rng
    def reference_request(self) -> RefRequest            # preview size and indexing
    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult
```

The spec contained two slightly different protocols (`compute_control(...)` in
§5.2 and `act(obs, ref)` in §5A.4). We use the §5A.4 shape (observation +
reference window in, result out) and add an explicit `RefRequest`:

| `RefRequest.mode` | Window index 0 | Spacing | Typical users |
|---|---|---|---|
| `"time"` | reference at the current time `t` | `dt` (default: control period) | LQR, iLQR, MPC, PID (time mode) |
| `"path"` | path point at the *estimated* monotonic progress `s₀` | `ds` metres (default 5 cm) | Pure Pursuit, PID (path mode) |

The runner, not the controller, owns progress. Path-mode progress is estimated
from the **observation** (so noise and latency affect it realistically) with a
forward-only search window, so it cannot jump across the figure-eight crossing.
The requested horizon is logged in each manifest, so differences in preview
are visible when comparing controllers.

`ControlResult` carries the command, a `Status` (`ok` / `fallback` /
`failed`), optional `predicted_states` / `predicted_controls` (logged and drawn in
animations) and scalar `diagnostics`. Solve time is measured by the runner.

Controllers are resolved by registry name, `package.module:Class`, or
`path/to/file.py:Class`.

## References: path vs trajectory

A `Path` is geometry only and comes from a **map**: `maps/csv/mapNN.csv` with
columns `s, x, y, yaw, kappa` on a 1 cm arclength grid, loaded by
`references.load_map`. Heading and curvature are written by the generator, so
the simulator never differentiates sampled points. Maps are produced by the
standalone `maps/generate.py` (or any tool writing the same columns); the
simulator does not know how a map was made.

A `Trajectory` = `Path` + speed profile, where the speed profile comes from the
scenario, so the same map can be driven under different limits. The speed
profile caps `v` by cruise speed, `v_max`, curvature
(`|ω_r| = v|κ| ≤ omega_fraction · ω_max`, leaving authority for feedback),
optional lateral acceleration, and longitudinal accel/decel via a
forward–backward pass. Optional stops create dwell segments.

Within each grid cell the speed is linear in time (constant acceleration), so
`ds/dt = v` holds exactly and `ω_r = v_r κ` by construction. `check_feasibility`
verifies the limits and the kinematic consistency of every scenario before it
runs.

## Simulation timing and event order

Time is an integer tick counter at `dt_physics` (default 5 ms). The control
period, sensor period and both latencies must be integer multiples of it
(validated). At each tick `k`:

1. **sensor sample** (sensor ticks): measure the current truth with
   bias + noise; dropout is drawn from a separate stream; queue for `k + L_s`.
2. **sensor delivery**: the newest due sample becomes the latest observation.
3. **control** (control ticks): build `Observation` + `ReferenceWindow`, call
   `compute` (wall-time measured, *not* simulated), limit (saturation + slew),
   queue the command for `k + L_u`.
4. **command delivery**: the newest due command becomes *eligible*.
5. **record + advance**: actuator lag (exact first-order) → applied velocity →
   slip / bias → kinematics → optional process noise.
6. **termination**: non-finite state, or path deviation above
   `max_path_deviation` for `deviation_grace` seconds (`tracking_failure`),
   otherwise `time_limit` at the end.

At `t = 0` the controller receives an initial (noisy) localisation fix, even if
the sensor has latency; this is a declared assumption.

Four latency-like effects are kept separate (spec §5A.5): sensor latency,
controller computation time (measured only), command latency, and actuator
lag. Deadline-emulation mode — mapping measured solve time to simulated delay
— is on the roadmap.

### Signal chain

```text
controller → raw ─▶ Limits ─▶ command ─▶ delay queue ─▶ eligible ─▶ actuator lag ─▶ applied ─▶ plant
               (logged)          (logged)                  (logged)                 (logged)
```

Logging both `raw` and limited commands means a controller that ignores
constraints shows up in the `constraint_violations` metric even though the
safety layer protected the plant.

## Randomness

`RngStreams(seed)` derives an independent generator per name
(`plant`, `sensor.noise`, `sensor.dropout`, `seed:controller`, …) from
`SeedSequence(seed, spawn_key=crc32(name))`. Noise is drawn even for dropped
samples, so changing the dropout rate does not change noise values, and
controllers never influence exogenous draws.

## Episode format

Each episode directory holds `manifest.json` (scenario dump, controller name +
config, seed, reference request, termination, config hash, versions, git
commit, metrics) and `episode.npz`:

| Rate | Arrays |
|---|---|
| physics tick | `t`, `truth`, `eligible`, `applied` |
| control tick | `ctrl_t`, `obs_pose`, `obs_t_sample`, `ref_pose`, `ref_control`, `ref_s`, `raw`, `command`, `solve_time`, `status`, `pred_states`?, `pred_controls`?, `diag_*` |
| sensor tick | `sensor_t`, `sensor_pose`, `sensor_dropped` |
| reference | `ref__*` (full trajectory, so nothing else is needed to plot or score) |

## Metrics (from logs, ground truth)

* **Time-indexed:** `pos_rmse/max/final`, `heading_rmse/mae`, `along_rmse`,
  `lateral_rmse`, `settling_time`.
* **Path:** `cte_rmse/max` (signed distance at monotonic progress), `completion`,
  `goal_distance`.
* **Control:** `effort`, `effort_deviation` (`∫‖u − u_r‖²`), `smoothness`.
* **Constraints:** `constraint_violations`, `constraint_violation_max` (raw output).
* **Compute:** `solve_ms_mean/p50/p95/p99/max`, `deadline_miss_rate`,
  `n_failed`, `n_fallback`.
* **Outcome:** `termination`, `success` (time limit reached, ≥ 99 % progress,
  ≤ 10 cm from the goal, no failed solves).

Time-indexed and path metrics answer different questions. Compare path
followers (Pure Pursuit) on `cte_*`, and trajectory trackers on `pos_*`.

## Benchmark suites

A suite YAML lists maps, scenarios (with `overrides` and cartesian `sweep`s
over dotted keys; a scenario entry may set its own `maps`), controllers (name,
label, config) and seeds. Every controller runs on every map × scenario variant
with the same seeds; without `maps` a scenario uses its default map. Output:

```text
runs/<stamp>_<suite>/
  suite.yaml  results.csv  summary.md  figures/
  episodes/<map>_<scenario>/<controller>/seed_<n>/{manifest.json, episode.npz}
```

`python visualize.py <dir>` aggregates (mean ± std over seeds) and draws bar charts
and sweep (degradation) curves.
