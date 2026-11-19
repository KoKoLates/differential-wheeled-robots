# Roadmap & progress

Status of the project against the [benchmark specification](differential_drive_control_benchmark.md).
Tick boxes as work lands; keep notes short and link PRs or results.

Legend: ✅ done · 🚧 in progress · ⬜ not started

## Infrastructure

### P1 — Core engine (M0a) ✅

- [x] Flat module layout run as scripts (`python main.py --arg ...`), pytest
- [x] Core types, geometry, limits, named RNG streams
- [x] Unicycle model: exact ZOH (stable small-ω branch), RK4, Euler; finite-difference Jacobians
- [x] References: arclength paths loaded from map files (`maps/`, standalone generator: shapes + random splines with a turn-radius check), speed profiles with curvature / accel caps and stops, timed trajectories, feasibility + consistency check
- [x] Monotonic progress tracking (robust at self-intersections)
- [x] `Controller` interface + registry (name, `module:Class`, `file.py:Class`)
- [x] Runner, episode log (NPZ + JSON manifest), metrics from logs
- [x] Static figures, animation (GIF/MP4, predicted horizon)
- [x] Scripts: `main.py` (run / list / show / `--save`), `visualize.py` (reports / compare / GIF)

### P2 — Experiment platform (M0b) ✅

- [x] Maps (geometry) separated from scenarios (conditions); scenario YAML with `base:` inheritance and dotted overrides; catalogue: nominal, recovery (I1, I2), stop-and-go (N5), noise / latency (R1), mismatch (R2)
- [x] Suites: map × scenario × controller × seed, cartesian sweeps, paired seeds, parallel jobs
- [x] `results.csv`, `summary.md`, aggregation (mean ± std), bar and sweep figures (`benchmark.py`, `visualize.py`)
- [x] Determinism and save/load/recompute tests

### P3 — Realism (M0c) ✅

- [x] Multi-rate integer-tick clock (physics / sensor / control), validated ratios
- [x] Sensor: noise, bias, dropout, latency (causal delivery queue)
- [x] Command pipeline: saturation, slew-rate limits, command latency; raw vs limited vs eligible vs applied logged
- [x] Plant: first-order actuator lag, slip, yaw-rate bias, process noise
- [x] Termination policy (tracking / numerical / controller errors)
- [x] Builtin robustness suites: `latency`, `noise`, `mismatch`

### Infrastructure backlog ⬜

- [ ] Deadline-emulation mode (map measured solve time to simulated delay)
- [ ] Wheel-speed limits (map `(v, ω)` to wheel speeds)
- [ ] State estimator plug-in slot (e.g. EKF between sensor and controller)
- [ ] Confidence intervals in reports; failure-category breakdown table
- [ ] Rigid-transform equivariance scenario family (G1) and near-π heading suite (G2)
- [ ] Replay: re-simulate from logged applied controls and verify truth
- [ ] Docs site (mkdocs)

## Phase A — classical & model-based controllers

| ID | Controller | Status | Notes |
|---|---|---|---|
| M1 | PID (time + path mode, anti-windup, filtered D) | ✅ baseline | `diffdrive/controllers/pid.py` |
| M1 | Pure Pursuit (arclength lookahead, adaptive L_d) | ✅ baseline | `diffdrive/controllers/pure_pursuit.py` |
| M2 | Time-varying LQR | ⬜ template | first implement `Unicycle.jacobians` |
| M3 | iLQR (receding horizon) | ⬜ template | |
| M4 | Linear MPC (OSQP) | ⬜ template | `pip install --group mpc` |
| M5 | Nonlinear MPC (CasADi multiple shooting) | ⬜ template | |
| M6 | SE(2) utilities | ⬜ template | `diffdrive/core/se2.py`, `tests/test_se2.py` |
| M6 | Lie-group LQR | ⬜ template | |
| M6 | Lie-group MPC | ⬜ template | |
| M7 | Phase A final benchmark & report | ⬜ | frozen test suite, multi-seed, timing tails |

For each controller (spec §5.4): derivation notes in `docs/derivations/`,
implementation, tests (the acceptance tests in
`tests/test_controller_acceptance.py` start running when `implemented = True`),
default config, and benchmark results.

### M1 checklist

- [x] P / PD / PID with anti-windup and derivative filter
- [x] Pure Pursuit with progress-based lookahead, adaptive lookahead
- [ ] Gain / lookahead sweeps (suite) and write-up: corner cutting, crossing behaviour
- [ ] Note: in the `latency` suite, Pure Pursuit's CTE *decreases* with latency. A hypothesis to verify: the delay counteracts its geometric corner cutting.

## Phase B — learning-based tracking (after M7)

| ID | Item | Status |
|---|---|---|
| L0 | Dataset export from episodes (`export_dataset.py`), episode-level splits, hold-out maps, `dataset` suite | ✅ |
| L0 | `PolicyController` (load a trained model, same features as the export) | ⬜ |
| L0 | Gymnasium adapter over the same runner (`step` = one control interval) | ⬜ |
| L1 | Behaviour cloning (MLP) | ⬜ |
| L2 | DAgger | ⬜ |
| L3 | PPO / SAC | ⬜ |
| L4 | Residual RL / learned-residual-dynamics MPC | ⬜ |
