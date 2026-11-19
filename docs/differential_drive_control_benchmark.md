# Differential-Drive Trajectory Tracking & Optimal Control Benchmark

> **Project 1 — Technical specification, learning roadmap, and reproducible experimental protocol**  
> Status: design specification · November 2026  
> Scope: differential-drive/unicycle robot, Phase A: eight classical/model-based controller families; Phase B: learning-based trajectory tracking; no obstacle avoidance or global planning.

> **Math rendering:** This file uses GitHub-flavored Markdown math: inline `$...$` and standalone `$$` blocks. In GitHub, view the rendered Markdown (not Raw). In Notion, Markdown import may not reliably convert LaTeX into equation blocks; if formulas appear as source text, paste each display equation into a Notion `/equation` block. Avoid pasting the entire Markdown into a plain text block.

## 0. Executive summary

**Research question:** Given the same feasible reference motion and differential-drive robot, how do classical feedback, geometric, linear optimal, nonlinear optimal, predictive, and Lie-group formulations compare in tracking accuracy, convergence, robustness, constraint satisfaction, and computation?

**Controller sequence:** PID → Pure Pursuit → trajectory-tracking LQR → iLQR → linear/time-varying MPC (QP) → nonlinear MPC (NLP) → Lie-group LQR → Lie-group MPC.

**Learning sequence:** Finish all eight classical controllers, including Lie-group LQR/MPC, before Phase B: BC → DAgger → PPO → SAC → Residual RL → Learning-based MPC. Small MLPs and CPU-scale experiments are sufficient for the first milestones; this is an experimental hypothesis, not a performance guarantee.

**Core deliverable:** a tested Python package, configuration-driven experiments, reproducible CSV/JSON results, trajectory and timing plots, GIF/MP4 animations, and a report explaining *why* controllers differ—not merely which scores lowest.

**Project boundary:** A separate Project 2 will handle global planning, local planning, obstacle avoidance, and trajectory optimization for collision-free navigation. iLQR in Project 1 deliberately teaches the mathematics of trajectory optimization, but obstacle avoidance is out of scope.

### Learning outcomes

1. Derive and discretize differential-drive kinematics, errors, and Jacobians.
2. Understand feedback vs feedforward, path vs time-indexed trajectory, and local vs global convergence.
3. Implement eight algorithms with consistent interfaces and well-defined assumptions.
4. Formulate quadratic costs, Riccati recursions, constrained QPs, nonlinear programs, and SE(2) error dynamics.
5. Design experiments that separate tracking quality, model mismatch, tuning, and optimization cost.
6. After completing Phase A, learn supervised imitation, interactive imitation, reinforcement learning, and learned dynamics on the same tracking task.

---

## 1. Problem formulation and assumptions

### 1.1 Robot state, inputs, and nonholonomic constraint

Use the kinematic unicycle abstraction of a differential-drive robot:

$$
\mathbf q = [x, y, \theta]^\top, \qquad \mathbf u = [v,\omega]^\top,
$$

$$
\dot{\mathbf q}=f(\mathbf q,\mathbf u)=
\begin{bmatrix}v\cos\theta\\v\sin\theta\\\omega\end{bmatrix}.
$$

The no-sideways-slip constraint is $-\sin\theta\,\dot x+\cos\theta\,\dot y=0$. The robot cannot directly command lateral velocity. Differential-drive wheel mapping (wheel radius $r$, wheel separation $b$):

$$
v=\frac r2(\dot\phi_R+\dot\phi_L),\qquad
\omega=\frac r b(\dot\phi_R-\dot\phi_L).
$$

This is a **kinematic** benchmark, not a full mass/inertia/torque dynamics study. Initially assume perfect tracking of velocity commands; later add actuator dynamics.

### 1.2 Discretization

For fixed $\Delta t$ and piecewise-constant $(v,\omega)$, the exact zero-order-hold (ZOH) step is

$$
\begin{aligned}
x_{k+1}&=x_k+\frac{v_k}{\omega_k}[\sin(\theta_k+\omega_k\Delta t)-\sin\theta_k],\\
y_{k+1}&=y_k-\frac{v_k}{\omega_k}[\cos(\theta_k+\omega_k\Delta t)-\cos\theta_k],\\
\theta_{k+1}&=\operatorname{wrap}(\theta_k+\omega_k\Delta t),
\end{aligned}
$$

when $|\omega_k|$ is not close to zero. As $\omega_k\to0$, use $x_{k+1}=x_k+v_k\Delta t\cos\theta_k$, $y_{k+1}=y_k+v_k\Delta t\sin\theta_k$. For stable code, use a numerically continuous sinc-based implementation or a small-$\omega$ expansion. Euler is fine for early unit tests; use exact ZOH or RK4 for the final benchmark. Preserve an **unwrapped heading internally** when useful, and wrap angular differences.

### 1.3 Reference path versus reference trajectory

A **path** $p(s)=[x(s),y(s)]$ is geometric; it does not prescribe when the robot reaches a point. A **trajectory** $\mathbf q_r(t),\mathbf u_r(t)$ includes timing and feedforward velocity. A valid reference must approximately satisfy $\dot x_r=v_r\cos\theta_r$, $\dot y_r=v_r\sin\theta_r$, $\dot\theta_r=\omega_r$.

For nonzero forward speed, path curvature $\kappa(s)=d\theta/ds$ implies $\omega_r=v_r\kappa$. Be careful near stops, cusps, and reversals. A geometric path is not automatically a dynamically/kinematically feasible timed trajectory.

**Two benchmark tracks are essential:**

- **Path-following track:** progress is free; measure cross-track distance and heading relative to closest/forward path location. Pure Pursuit is naturally suited to this.
- **Time-indexed trajectory-tracking track:** compare pose at the prescribed $t$ and use $v_r(t),\omega_r(t)$ as feedforward. LQR, iLQR, and MPC naturally fit here.

Do not claim a single overall winner across these tracks without documenting each controller's available reference information and speed policy.

### 1.4 Error conventions

World-frame error: $\mathbf e_w=[x_r-x,y_r-y,\operatorname{wrap}(\theta_r-\theta)]^\top$.

Body-frame error:

$$
\begin{bmatrix}e_x\\e_y\\e_\theta\end{bmatrix}=
\begin{bmatrix}\cos\theta&\sin\theta&0\\-\sin\theta&\cos\theta&0\\0&0&1\end{bmatrix}
\begin{bmatrix}x_r-x\\y_r-y\\\operatorname{wrap}(\theta_r-\theta)\end{bmatrix}.
$$

For this convention, the continuous error dynamics are

$$
\dot e_x=\omega e_y-v+v_r\cos e_\theta,\quad
\dot e_y=-\omega e_x+v_r\sin e_\theta,\quad
\dot e_\theta=\omega_r-\omega.
$$

**Important:** These body-frame translational errors are *not generally equal* to the translational coordinates of $\log(T_r^{-1}T)$. Also, choose one error sign convention throughout and test it.

### 1.5 Constraints and optional actuator lag

Initial example limits (adjust after baseline tuning):

$$
0\le v\le1.0\ \mathrm{m/s},\qquad |\omega|\le1.5\ \mathrm{rad/s}.
$$

Rate limits: $|v_k-v_{k-1}|\le a_{\max}\Delta t$, $|\omega_k-\omega_{k-1}|\le\alpha_{\max}\Delta t$. These are command slew limits, not necessarily wheel torque constraints. If wheel limits are modeled, map $(v,\omega)$ to wheel speeds and enforce those too.

Optional plant state $[x,y,\theta,v,\omega]$ with commands $[v_c,\omega_c]$:

$$
\dot v=(v_c-v)/\tau_v,\qquad \dot\omega=(\omega_c-\omega)/\tau_\omega.
$$

Use the simple model first; introduce lag as a controlled model-mismatch experiment.

---

## 2. Eight controller deep dives

### 2.1 PID — classical error feedback

**Problem solved.** Correct heading/speed deviations with a cheap feedback law, without requiring a predictive robot model. A naive heading PID alone does **not** guarantee lateral path convergence.

$$
e_\theta=\operatorname{wrap}(\theta_{\mathrm{target}}-\theta),\quad
\omega_c=K_pe_\theta+K_i\int e_\theta dt+K_d\dot e_\theta.
$$

Choose a target heading that incorporates lateral error, e.g. $\theta_{\mathrm{target}}=\theta_{\mathrm{path}}+\arctan(k_y e_{\mathrm{lateral}}/\max(v,\epsilon))$ with a carefully defined sign convention. Alternatively, use separate body-frame error feedback. Control $v$ with feedforward/reference tracking or a separate speed loop. Derivative filtering and integrator anti-windup matter.

**Properties:** reactive, local, no explicit lookahead, no native constraint handling, negligible compute.

**Pros:** easy to implement/debug; strong baseline; inexpensive; often adequate on smooth paths.

**Cons:** sensitive to gain tuning; integrator windup; derivative noise; no anticipation of upcoming turns; coupling between lateral and heading error must be designed rather than assumed.

**Best tests:** step heading change, straight-to-turn transition, noise, saturation, windup, settling time.

**Implementation tasks:** P → PD → PID; anti-windup; derivative-on-measurement/filter; separate longitudinal policy; gain sweep.

### 2.2 Pure Pursuit — geometric path following

**Problem solved.** Follow a geometric path by steering toward a lookahead target. For a unicycle, compute angular speed rather than an Ackermann steering angle.

Let the lookahead point be $(x_L,y_L)$ in robot body coordinates and $L_d=\sqrt{x_L^2+y_L^2}$. The circular-arc curvature is

$$
\kappa_c=\frac{2y_L}{L_d^2},\qquad \omega_c=v_c\kappa_c.
$$

Typically select the target **ahead in path arclength** from a progress/closest-point estimate, not by an arbitrary closest Euclidean point that may jump across a figure-eight intersection. Consider $L_d=L_0+k_v|v|$.

**Properties:** geometric, intuitive lookahead, speed coupled to turning rate, no full dynamics model.

**Pros:** simple, stable-looking behavior on smooth paths, anticipatory compared with heading-only PID, little compute.

**Cons:** corner cutting, lookahead trade-off (short = oscillation; long = lag), ambiguous at self-intersections, no native input/rate constraints, no timed arrival guarantee.

**Best tests:** curvature sweep, figure-eight intersection, lookahead sensitivity, low-speed behavior, forward/reverse cases (reverse requires a deliberate design).

**Implementation tasks:** path arclength parameterization; progress tracking; body-frame target transform; fixed/adaptive lookahead; speed policy.

### 2.3 Trajectory-tracking LQR — local linear optimal feedback

**Problem solved.** Choose feedback gains minimizing quadratic tracking error and control effort around a feasible nominal trajectory.

Linearize the discrete dynamics about $(\bar q_k,\bar u_k)$:

$$
\delta q_{k+1}=A_k\delta q_k+B_k\delta u_k.
$$

For first-order Euler discretization:

$$
A_k\approx I+\Delta t\begin{bmatrix}0&0&-\bar v_k\sin\bar\theta_k\\0&0&\bar v_k\cos\bar\theta_k\\0&0&0\end{bmatrix},\quad
B_k\approx\Delta t\begin{bmatrix}\cos\bar\theta_k&0\\\sin\bar\theta_k&0\\0&1\end{bmatrix}.
$$

Use Jacobians of the **actual discrete integrator** in the final implementation. Minimize

$$
J=\delta q_N^\top Q_f\delta q_N+\sum_{k=0}^{N-1}(\delta q_k^\top Q\delta q_k+\delta u_k^\top R\delta u_k).
$$

Finite-horizon Riccati recursion:

$$
K_k=(R+B_k^\top P_{k+1}B_k)^{-1}B_k^\top P_{k+1}A_k,
$$

$$
P_k=Q+A_k^\top P_{k+1}A_k-A_k^\top P_{k+1}B_kK_k,
$$

with $P_N=Q_f$ and $u_k=\bar u_k-K_k\delta q_k$. In code, solve linear systems rather than explicitly inverting matrices.

**Critical caveat:** The unicycle linearized about **stationary** $v_r=\omega_r=0$ is not fully controllable in lateral position; do not claim a fixed LQR around arbitrary stopped poses globally stabilizes the robot. Use trajectory-tracking/time-varying LQR around moving references and state its local validity.

**Properties:** model-based, local, quadratic optimality for linearized model, low online compute once gains are available, no inherent hard constraints.

**Pros:** interpretable $Q/R$ trade-off; systematic gain design; efficient; excellent foundation for iLQR/MPC.

**Cons:** depends on reference and linearization; large-error limitations; no direct hard input constraints; careful angle handling required.

**Best tests:** small vs large initial errors, curvature/speed sweep, linearization error, saturation, fixed vs time-varying gains.

### 2.4 iLQR — iterative nonlinear trajectory optimization

**Problem solved.** Optimize a finite sequence of controls for nonlinear dynamics, using repeated local quadratic approximations. Unlike plain LQR, the nominal trajectory itself is updated.

$$
\min_{u_{0:N-1}}\ \ell_f(q_N)+\sum_{k=0}^{N-1}\ell(q_k,u_k),\quad q_{k+1}=f_d(q_k,u_k).
$$

At each iteration: (1) rollout nominal $q,u$; (2) compute dynamics derivatives $f_x,f_u$ and cost derivatives; (3) backward dynamic-programming pass; (4) forward line-search rollout; (5) accept improvement and repeat.

Using local quadratic action-value terms $Q_x,Q_u,Q_{xx},Q_{ux},Q_{uu}$:

$$
k_k=-Q_{uu}^{-1}Q_u,\qquad K_k=-Q_{uu}^{-1}Q_{ux},
$$

$$
u_k^{\mathrm{new}}=\bar u_k+\alpha k_k+K_k(q_k^{\mathrm{new}}-\bar q_k).
$$

This is the **iLQR/Gauss–Newton-style** approximation: second derivatives of the dynamics are neglected in the value expansion (full DDP includes them). Add regularization, line search, and convergence checks. Standard iLQR does **not automatically enforce hard constraints**; clipping controls changes the problem and can degrade convergence. Constrained iLQR variants exist.

**Properties:** nonlinear dynamics, iterative local optimization, warm-start sensitive, generally local minima, feedback gains along nominal trajectory.

**Pros:** teaches optimal control deeply; often efficient relative to generic NLP; produces feedforward + local feedback policy; reusable for receding-horizon control.

**Cons:** derivative and numerical complexity; nonconvex/local convergence; constraint handling requires extra machinery; offline full-horizon and online replan timings are not directly comparable.

**Best tests:** initialization sensitivity, horizon length, regularization, convergence trace, nominal rollout vs disturbed feedback execution.

**Important comparison:** iLQR is an **optimizer**; MPC is a **receding-horizon control strategy**. iLQR can be used *inside* nonlinear MPC.

### 2.5 Linear MPC — constrained receding-horizon QP

**Problem solved.** Repeatedly optimize future controls under linear(ized) dynamics and linear constraints, apply the first control, then re-solve.

At time $t$, linearize around a nominal/reference trajectory. In deviation coordinates, when the nominal is dynamically consistent:

$$
\delta q_{k+1}=A_k\delta q_k+B_k\delta u_k.
$$

If it is not dynamically consistent, include the affine defect term $d_k=f_d(\bar q_k,\bar u_k)-\bar q_{k+1}$. Solve

$$
\min\sum_{k=0}^{N-1}\big(\|\delta q_k\|_Q^2+\|\delta u_k\|_R^2+\|\Delta u_k\|_S^2\big)+\|\delta q_N\|_{Q_f}^2
$$

subject to linearized dynamics, input bounds, rate bounds, and optionally linearized state constraints. With PSD quadratic objective and linear constraints, this is a convex QP; feasibility and solver termination must still be checked.

**Properties:** predictive, constrained, replan each step, computationally moderate, local model validity.

**Pros:** explicit hard linear constraints; practical OSQP solvers; warm starts; interpretable trade-offs.

**Cons:** linearization error; horizon/tuning dependence; infeasible QPs possible; large-angle deviations can be problematic.

**Best tests:** input saturation, slew limits, large initial heading error, horizon and control-rate sweeps, QP solve-time tails.

**Implementation tasks:** sparse QP matrices; correct initial-state constraint; feedforward + deviations; affine defects; warm start; infeasibility/fallback handling; log solver status.

### 2.6 Nonlinear MPC — constrained nonlinear predictive control

**Problem solved.** Re-solve a finite-horizon constrained optimization while retaining nonlinear dynamics:

$$
\min_{q_{0:N},u_{0:N-1}}\ \ell_f(q_N)+\sum_{k=0}^{N-1}\ell(q_k,u_k)
$$

$$
\text{s.t.}\quad q_0=\hat q_t,\quad q_{k+1}=f_d(q_k,u_k),\quad u_k\in\mathcal U,\quad \Delta u_k\in\mathcal D.
$$

Use multiple shooting in CasADi + IPOPT initially. This is an **NLP**, not a globally convex optimization problem. SQP or interior-point algorithms generally return locally optimal/feasible solutions subject to solver conditions.

**Properties:** nonlinear prediction, native nonlinear constraints, repeated optimization, warm-start sensitive.

**Pros:** more faithful model over large turns; expressive constraints; strong reference method for predictive control.

**Cons:** higher compute and variable latency; convergence/infeasibility issues; initialization and solver tuning matter; model mismatch still exists.

**Best tests:** high-curvature reference, large pose offsets, nonlinear vs linear prediction error, timing P95/P99, solver failures.

**Fair comparison:** keep horizon, stage costs, bounds, integration method, and stopping tolerances as aligned as practical with Linear MPC; report differences that cannot be matched.

### 2.7 Lie-group LQR — geometric tracking error and local optimal feedback

**Problem solved.** Design local optimal feedback using pose error on $SE(2)$ rather than naively subtracting global pose coordinates.

$$
T=\begin{bmatrix}R(\theta)&p\\0&1\end{bmatrix}\in SE(2),\qquad
\hat\xi=\begin{bmatrix}0&-\omega&v_x\\\omega&0&v_y\\0&0&0\end{bmatrix}.
$$

For a unicycle body twist $\xi=[v,0,\omega]^\top$, $\dot T=T\hat\xi$. One choice of relative error is

$$
E=T_r^{-1}T,\qquad e=\operatorname{Log}(E)^\vee\in\mathbb R^3.
$$

Derive the corresponding error dynamics, linearize them in local algebra coordinates, and compute time-varying LQR gains. Explicitly document left- vs right-invariant error conventions and how perturbations are transported (adjoint/Jacobians). A Lie algebra is a **local vector space**, not a promise of global linearity.

**Properties:** coordinate-aware geometry, local logarithmic errors, same Riccati machinery after proper linearization.

**Pros:** principled pose-error treatment; natural group composition; useful invariance/equivariance tests; foundation for SE(3) robotics.

**Cons:** Exp/Log, adjoints, Jacobians, sign conventions add complexity; local singularity/branch behavior around angle wrapping; not automatically more accurate than a well-designed body-frame LQR.

**Best tests:** rigid world-frame transforms, large heading offsets, near-$\pi$ orientation differences, Euclidean vs body-frame vs log errors.

**Key experimental insight:** a conventional body-frame tracking error is already geometrically meaningful. Lie-group improvements must be demonstrated, not presumed.

### 2.8 Lie-group MPC — predictive optimization on SE(2)

**Problem solved.** Predict and optimize pose evolution with group-consistent updates and/or invariant errors.

A typical group-valued discrete update is

$$
T_{k+1}=T_k\operatorname{Exp}(\Delta t\,\hat\xi(u_k)).
$$

An objective might penalize

$$
e_k=\operatorname{Log}(T_{r,k}^{-1}T_k)^\vee,\quad
J=\sum_k e_k^\top Qe_k+\delta u_k^\top R\delta u_k.
$$

Choose either (a) a **Lie-group linearized MPC/QP** using local error dynamics, or (b) a **nonlinear Lie-group MPC/NLP** with group-consistent integration. These are *different* implementations; specify which is being compared. For this project, implement (a) first, then optionally (b). Avoid confusing improved integration with improved error coordinates: test these factors separately.

**Properties:** predictive constraints + geometric state representation; still finite-horizon and generally local/nonconvex in nonlinear formulations.

**Pros:** group-consistent integration, coordinate-frame clarity, transferable techniques for SE(3) estimation/control.

**Cons:** hardest to derive/debug; numerical overhead; benefits can be small for simple planar tracking; no guarantee of global optimality or stability.

**Best tests:** rigid-transform equivariance, heading wrap, high-curvature paths, matched-error/cost ablations, compute overhead.

---

## 3. Comparison matrix

| Controller | Model required | Lookahead/prediction | Hard constraints native? | Optimization | Typical computation | Main limitation |
|---|---|---|---|---|---|---|
| PID | No full dynamics | Reactive | No | None | Very low | No explicit anticipation |
| Pure Pursuit | Geometric unicycle relation | Spatial lookahead | No | None | Very low | Corner cutting/lookahead tuning |
| LQR | Linearized dynamics | Finite-horizon design / gain schedule | No | Riccati recursion | Low online | Local linear validity |
| iLQR | Nonlinear dynamics + derivatives | Finite horizon | Not in standard form | Iterative local optimal control | Medium–high | Local convergence/constraints |
| Linear MPC | Linearized dynamics | Receding horizon | Yes, linear bounds | QP | Medium | Model approximation |
| Nonlinear MPC | Nonlinear dynamics | Receding horizon | Yes, modeled bounds | NLP | High/variable | Solver latency/local minima |
| Lie-group LQR | SE(2) error dynamics | Gain schedule | No | Geometric linearization + Riccati | Low–medium | Derivation complexity/locality |
| Lie-group MPC | SE(2) dynamics/errors | Receding horizon | Formulation-dependent | QP or NLP | Medium–high | Complexity/limited guaranteed gain |

**Never interpret this table as an unconditional ranking.** All methods require tuning and can fail outside their assumptions.

---

## 4. Reproducible benchmark protocol

### 4.1 Experimental factors and controlled variables

**Keep constant per matched run:** robot plant, integration method, reference, starting pose, simulation duration, command limits, observation stream, control frequency, disturbance seed, and evaluation code. Fix compute hardware/software versions and solver tolerances for timing studies.

**Vary intentionally:** controller, controller parameters, reference curvature/speed, initial error, sensor noise, actuator lag, control bounds, model mismatch, horizon length, and solver tolerance.

**Tuning policy:** allocate each controller a documented validation set and similar tuning budget (manual trials or automated search); freeze parameters before the held-out test set. Avoid selecting best parameters using test outcomes.

**Two modes:** (1) ideal state-feedback/no-lag baseline; (2) estimated noisy state/lag/disturbance. State explicitly whether noise affects measurement only or plant evolution.

### 4.2 Scenario suite

| ID | Scenario | Controlled sweep | Question answered |
|---|---|---|---|
| N1 | Straight line | speed, distance | Baseline tracking and steady-state error |
| N2 | Circle | radius, speed | Curvature tracking and feedforward |
| N3 | Figure-eight | speed, crossing geometry | Curvature reversal and progress ambiguity |
| N4 | S-curve | curvature change rate | Anticipation vs reactive correction |
| N5 | Start–stop | speed profile | Timing, acceleration/rate constraints |
| I1 | Initial lateral offset | e.g. 0–2 m | Region of attraction/recovery |
| I2 | Initial heading offset | e.g. 0–180° | Large-angle recovery and wrapping |
| R1 | Pose noise | position/heading std dev | Noise sensitivity |
| R2 | Actuator lag | $\tau_v,\tau_\omega$ | Model mismatch |
| R3 | Velocity bias/slip | bias magnitude | Disturbance rejection |
| C1 | Input saturation | max $v,\omega$ | Constraint behavior |
| C2 | Input slew limits | max $\Delta v,\Delta\omega$ | Smoothness/feasibility |
| P1 | Horizon sweep | $N,\Delta t$ | Accuracy/compute trade-off |
| P2 | Solver tolerances | tolerance, max iterations | Latency/reliability |
| G1 | Rigid frame transform | rotate/translate whole world | Geometric equivariance |
| G2 | Near-$\pi$ heading | angle offsets | Log-map branch sensitivity |

Suggested *starting* values are not physical truth: $\Delta t_{plant}=0.01\,s$, $\Delta t_{control}=0.05\,s$, $v_{max}=1\,m/s$, $|\omega|_{max}=1.5\,rad/s$. Verify reference feasibility and adjust to avoid designing impossible tests. Run multiple seeds (e.g. 20–50) for stochastic conditions and report uncertainty.

### 4.3 Metric definitions

Let $e_{p,k}=\|p_k-p_{r,k}\|_2$ for time-indexed tracking. For path following, use signed/unsigned cross-track distance to a geometrically associated path point and **label it differently**. Let $e_{\theta,k}=\operatorname{wrap}(\theta_k-\theta_{r,k})$.

| Metric | Definition | Interpretation / caveat |
|---|---|---|
| Position RMSE | $\sqrt{\frac1N\sum e_{p,k}^2}$ | Timed position accuracy; not interchangeable with cross-track RMSE |
| Max position error | $\max_k e_{p,k}$ | Worst excursion; sensitive to one-off anomalies |
| Heading MAE/RMSE | $\frac1N\sum|e_{\theta,k}|$, $\sqrt{\frac1N\sum e_{\theta,k}^2}$ | Always wrap angle differences |
| Cross-track RMSE | RMSE of signed normal distance to path | Path-following accuracy, not schedule accuracy |
| Along-track / phase error | Tangential distance or progress difference | Separates lateness from lateral tracking |
| Control effort | $\sum_k (w_v v_k^2+w_\omega\omega_k^2)\Delta t$ | Specify units/weights; raw effort may penalize necessary forward motion |
| Control deviation effort | $\sum_k\|u_k-u_{r,k}\|_R^2\Delta t$ | More comparable for timed trajectories |
| Control smoothness | $\sum_k\|u_k-u_{k-1}\|_S^2$ | State whether divided by $\Delta t$; normalize if comparing rates |
| Settling time | First time error enters **and stays** inside tolerances | Only for appropriate fixed/steady targets; define thresholds/window |
| Constraint violation | Count, max magnitude, and duration above bounds | Record raw command, applied command, plant state separately |
| Success / failure | Meets terminal/trajectory tolerance without safety or solver failure | Define objective threshold before running experiments |
| Runtime | mean, median, P95, max per control step | Also log solver-only, wall time, initialization, and warm-start status |
| Deadline miss rate | fraction of steps exceeding control period | Critical for real-time suitability |
| Optimization convergence | objective decrease, iterations, status | Especially iLQR/NMPC; do not hide failed solves |

**Timing detail:** P95 is the 95th percentile of measured durations; it does not mean the maximum. Log both. Consider P99 if sufficient samples exist. Do not combine offline precomputation (LQR gains or full-horizon iLQR) with online per-step solve time without separating them.

**Uncertainty:** report mean/median, standard deviation or interquartile range, and confidence intervals across seeds. Avoid ranking methods on tiny differences smaller than run-to-run variability.

### 4.4 Failure and safety policy

A run fails if any **predeclared** condition holds, e.g. nonfinite state/control, solver infeasible without recovery, deviation above a specified threshold for a specified duration, failure to reach terminal tolerance within time limit, or a deadline miss if hard real-time execution is the test. Distinguish `tracking_failure`, `solver_failure`, `constraint_failure`, and `deadline_failure`.

A shared actuator limiter may clip commands for safety; log both pre-clamp and post-clamp commands. Otherwise clipping can hide the fact that a controller ignores constraints.

### 4.5 Ablations and fairness

1. **Error coordinates:** world subtraction vs body-frame error vs SE(2) log error.
2. **Dynamics integrator:** Euler vs RK4 vs exact ZOH, holding other variables fixed.
3. **Horizon:** $N$ and $N\Delta t$ separately; same number of steps does not imply same prediction time if step sizes differ.
4. **Warm start:** cold vs warm solve; log iterations and solve time.
5. **Cost weights:** comparable physical units and scaling; no arbitrary same numeric $Q/R$ across different coordinate conventions.
6. **Reference information:** path-only and timed-trajectory tests reported separately.
7. **Lie-group benefit:** distinguish error-coordinate effects, integrator effects, and solver effects; do not change all three simultaneously.

### 4.6 Expected artifacts per run

- `config.yaml` (all parameters and seed)
- `trajectory.csv` (time, reference, actual/estimated pose, errors, raw/applied commands)
- `solver.csv` (solve time, iterations, status, objective, constraint residual)
- `metrics.json` (metrics and failure category)
- `path.png`, `errors.png`, `controls.png`, `timing.png`
- `animation.mp4` or `.gif` (robot, reference, predicted horizon when applicable)

---

## 5. Architecture and code templates

### 5.1 Repository layout

```text
control-benchmark/
├── pyproject.toml
├── README.md
├── configs/
│   ├── robot.yaml
│   ├── controllers/
│   └── experiments/
├── src/control_benchmark/
│   ├── core/
│   │   ├── types.py
│   │   ├── interfaces.py
│   │   └── geometry.py
│   ├── models/
│   │   ├── unicycle.py
│   │   └── actuator_lag.py
│   ├── references/
│   │   ├── trajectory.py
│   │   └── generators.py
│   ├── controllers/
│   │   ├── pid.py
│   │   ├── pure_pursuit.py
│   │   ├── lqr.py
│   │   ├── ilqr.py
│   │   ├── linear_mpc.py
│   │   ├── nonlinear_mpc.py
│   │   ├── lie_lqr.py
│   │   └── lie_mpc.py
│   ├── solvers/
│   │   ├── qp.py
│   │   └── nlp.py
│   ├── simulation/
│   │   ├── runner.py
│   │   ├── sensors.py
│   │   └── disturbances.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── visualization.py
│   └── lie/
│       └── se2.py
├── tests/
│   ├── test_dynamics.py
│   ├── test_jacobians.py
│   ├── test_references.py
│   ├── test_se2.py
│   └── test_controllers.py
├── experiments/
├── notebooks/
└── docs/
    ├── derivations/
    └── benchmark_protocol.md
```

### 5.2 Data types and controller protocol

```python
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

@dataclass(frozen=True)
class RobotState:
    x: float
    y: float
    theta: float

@dataclass(frozen=True)
class ControlCommand:
    v: float
    omega: float

@dataclass(frozen=True)
class ReferenceState:
    t: float
    x: float
    y: float
    theta: float
    v: float
    omega: float

@dataclass
class ControlResult:
    command: ControlCommand
    solve_time_s: float
    status: str
    predicted_states: Array | None = None  # (N+1, 3)
    predicted_controls: Array | None = None  # (N, 2)
    iterations: int | None = None

class ReferenceTrajectory(Protocol):
    def sample(self, t: float) -> ReferenceState: ...
    def sample_horizon(self, t: float, n: int, dt: float) -> list[ReferenceState]: ...
    def closest_progress(self, state: RobotState, previous_progress: float) -> float: ...

class Controller(Protocol):
    def reset(self) -> None: ...
    def compute_control(
        self,
        state_estimate: RobotState,
        reference: ReferenceTrajectory,
        t: float,
        previous_applied_command: ControlCommand,
    ) -> ControlResult: ...
```

`closest_progress` must be monotonic or otherwise robust to figure-eight crossings. Do not expose simulator ground truth to a noisy-state controller; the simulator passes an estimate. For timed tracking, use `sample(t)` or `sample_horizon(...)`.

### 5.3 Minimal simulator skeleton

```python
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SimConfig:
    plant_dt: float = 0.01
    control_dt: float = 0.05
    duration_s: float = 20.0
    v_min: float = 0.0
    v_max: float = 1.0
    omega_max: float = 1.5

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def step_unicycle_euler(s: RobotState, u: ControlCommand, dt: float) -> RobotState:
    return RobotState(
        x=s.x + dt * u.v * math.cos(s.theta),
        y=s.y + dt * u.v * math.sin(s.theta),
        theta=wrap_angle(s.theta + dt * u.omega),
    )

# TODO: implement exact-ZOH integration, command limiting, observation noise,
# separate plant/control clocks, run logging, and deterministic seeds.
```

This is an **educational skeleton**, not a complete benchmark simulator. The final runner should support zero-order-hold commands between controller ticks, integer-related or event-scheduled plant/control steps, and explicit separation of plant state, observations, and reference.

### 5.4 Controller development checklist

For each controller, create: `derivation.md`, `implementation.py`, `test_*.py`, `default.yaml`, and `results/`. Record assumptions, exact equations, tuning parameters, reference access, numerical method, convergence/failure behavior, and benchmark findings.

### 5.5 Suggested dependencies

Python 3.11+, NumPy, SciPy, Matplotlib, pandas, PyYAML, pytest; OSQP and `scipy.sparse` for QP; CasADi + IPOPT for NLP; optionally JAX for derivative validation and Plotly for interactive analysis. Pin versions in the lockfile and report CPU/OS/Python versions for timing comparisons. Avoid ROS 2 and heavyweight physics simulators in the first iteration.

---


## 5A. Simulator-first engineering blueprint — build once, reuse everywhere

> **Architectural requirement:** Build and validate this environment in M0 **before** implementing PID. Every later algorithm (classical or learned) plugs into the same controller API, runner, reference provider, data logger, and benchmark evaluator. Do not build a separate RL simulator. Gymnasium should be an *adapter* over the same core engine.

### 5A.1 Goals, non-goals, and design invariants

The simulator must support (1) deterministic offline closed-loop evaluation; (2) open-loop rollouts and expert-labeled synthetic datasets; (3) seeded sensor noise, bias, dropout, and observation latency; (4) command/network latency, sample-and-hold, actuator lag, saturation, and rate limits; (5) configurable model mismatch, slip, and disturbances; (6) multi-rate physics/sensor/controller clocks; (7) scenario sweeps, replay, metrics, plots, and train/validation/test isolation; (8) a Gymnasium adapter for RL.

**Do not overbuild:** no ROS 2, collision engine, map server, distributed execution, differentiable physics engine, or plugin registry in M0. Use plain Python classes, dataclasses, `Protocol`s, NumPy, and a small number of well-defined extension points.

**Invariants:**

- The controller **never** receives privileged ground-truth state unless a scenario explicitly declares an oracle-information experiment.
- A controller sees only an `Observation` and `ReferenceWindow`, and outputs a `ControlCommand`; it does not directly mutate simulator state.
- Simulation time is deterministic and **separate** from wall-clock solve time; wall-clock measurements are logged, not used as simulated delay unless an explicit real-time emulation mode is enabled.
- Physics integration is the single source of truth; the controller's internal prediction model can intentionally differ from it.
- All randomness comes from independent seeded generators (scenario, dynamics, sensors, training); never use an implicit global RNG.
- Each episode stores exact configuration, code revision, seed, split, trajectory ID, and controller version.
- Changing a controller must not require editing the simulation loop, metric formulas, or benchmark definitions.

### 5A.2 Layered architecture and data flow

```text
ScenarioConfig + Seed + ReferenceGenerator
                  |
                  v
             EpisodeRunner <----------------- BenchmarkSuite
                  |
       +----------+----------+
       |                     |
       v                     v
  TruePlant             Clock / EventQueue
  (truth state)         (multi-rate timing)
       |                     |
       +----> SensorModel ---+----> ObservationBuffer
       |                                |
       |                         delayed observation
       |                                v
       |                         ControllerAdapter
       |                        + ReferenceWindow
       |                                |
       |                          ControlCommand
       |                                v
       +<---- ActuatorModel <---- CommandDelayQueue
       |
       v
   EpisodeRecorder ---> DatasetExporter / Replay / Metrics / Plots
                              |
                         GymnasiumAdapter
                         (same runner core)
```

**Separation of concerns:** `TruePlant` integrates physical state; `SensorModel` emits noisy/delayed measurements; `ObservationBuffer` models availability and timestamps; `Controller` decides actions; `ActuatorModel` converts commanded to applied controls; `Recorder` captures both observable and privileged data; `Evaluator` computes scores *after* rollout from the saved trace.

### 5A.3 Proposed repository structure (extends Section 5.1)

```text
src/control_benchmark/
  core/
    types.py                 # State, Action, Observation, ReferenceWindow, StepResult
    protocols.py             # Controller, Dynamics, Sensor, Actuator contracts
    rng.py                   # independent seeded RNG streams
    config.py                # validated configs + schema/version
  simulation/
    clock.py                 # integer-tick scheduler; no float-time drift
    plant.py                 # true nonlinear kinematics and optional dynamics
    integrators.py           # exact ZOH / RK4 / Euler for tests
    sensors.py               # measurement, noise, dropout, timestamp
    delay.py                 # timestamped delivery queues
    actuators.py             # limits, slew rate, first-order lag
    disturbances.py          # slip, biases, external perturbations
    runner.py                # single episode orchestration
    gym_env.py               # thin RL adapter, not a second simulator
  references/
    generators.py            # feasible motion profiles, seeded families
    trajectory.py            # time interpolation, horizon queries
  evaluation/
    scenarios.py             # frozen scenario catalog
    suite.py                 # matrix expansion + paired seeds
    metrics.py               # pure functions over traces
    recorder.py              # truth, observation, command, applied, events
    exporter.py              # Parquet/NPZ datasets + manifest
    replay.py                # replay actions and validate determinism
    plots.py                 # trajectories, errors, timing, distributions
  learning/
    datasets.py              # PyTorch dataset + normalization
    expert.py                # expert action queries / quality filters
    bc.py
    dagger.py
    rl.py
    dynamics.py
configs/
  sim/base.yaml
  scenarios/nominal.yaml
  scenarios/noise_latency.yaml
  suites/phase_a_v1.yaml
  suites/learning_ood_v1.yaml
  splits/v1.yaml
results/                     # gitignored; keep manifests in reports
```

### 5A.4 Minimal contracts (stable from M0)

```python
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]

@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float

@dataclass(frozen=True)
class Control:
    v: float
    omega: float

@dataclass(frozen=True)
class Observation:
    pose: Pose2D                 # estimated, never implicitly truth
    t_sample: float             # when the sensor sampled
    t_received: float           # when available to controller
    previous_applied: Control   # only if exposed by observation config

@dataclass(frozen=True)
class ReferenceWindow:
    times: Vec                  # absolute simulation timestamps
    poses: Vec                  # shape (H+1, 3)
    controls: Vec               # shape (H+1, 2), feasible feedforward

@dataclass(frozen=True)
class ControllerResult:
    command: Control
    status: str = "ok"
    diagnostics: dict = field(default_factory=dict)

class Controller(Protocol):
    def reset(self, *, seed: int) -> None: ...
    def act(self, obs: Observation, ref: ReferenceWindow) -> ControllerResult: ...

class Plant(Protocol):
    def reset(self, pose: Pose2D) -> None: ...
    def advance(self, applied: Control, dt: float, rng: np.random.Generator) -> Pose2D: ...

class Sensor(Protocol):
    def sample(self, truth: Pose2D, t: float, rng: np.random.Generator) -> Observation | None: ...
```

**Implementation notes:** These are *interfaces*, not a complete runnable engine. The runner stamps `t_received` after applying delivery latency; a real implementation may separate `SensorSample` from delivered `Observation`. `previous_applied` is privileged unless a realistic actuator-feedback channel is configured; otherwise replace it with last commanded control or an optional field. For delayed measurements, the controller must know whether `pose` refers to sample time or has been propagated to current time. Future reference information must be consistently budgeted across controllers.

### 5A.5 Simulation clocks, causality, and latency

Use an integer base tick (e.g., `dt_physics = 0.005 s`) and schedule events at integer multiples of that tick. Example: physics 200 Hz, sensors 50 Hz, control 20 Hz; enforce integral ratios initially. The physics state advances at each base tick; sample events read the *current* truth; delivery events make measurements available; control events use the latest **delivered** observation; commands enter a delay queue; actuator state updates; recorder logs timestamps. Document the exact event ordering for events sharing a timestamp.

Distinguish four separate phenomena:

1. **Sensor latency:** measurement taken at $t_s$ arrives at $t_s+L_s$; never rewrite its timestamp to arrival time.
2. **Controller computation time:** measured wall-clock runtime; **not** automatically a simulation delay. In explicit deadline-emulation experiments, map it to simulated availability using a declared deterministic rule, otherwise keep timing metrics separate.
3. **Command/communication latency:** command generated at $t_c$ becomes eligible for application at $t_c+L_u$; queue and hold the previous command until then.
4. **Actuator lag:** applied velocity follows the eligible command via first-order dynamics, e.g. $\dot v=(v_{cmd}-v)/\tau_v$ and $\dot\omega=(\omega_{cmd}-\omega)/\tau_\omega$.

For nonzero actuator lag, augment the **true plant** state with actual velocities $(v,\omega)$; pose-only state remains the basic ideal model. If actuator feedback is unavailable, these states are hidden from the policy. Never pretend command velocity equals applied velocity under lag.

**Example timing:** controller at 20 Hz ($50$ ms), observation latency $30$ ms, command latency $50$ ms, physics at $5$ ms. A sample captured at $t=0.10$ may become visible at $t=0.13$; a control event at $t=0.15$ can use it; a command produced at $t=0.15$ becomes eligible at $t=0.20$. This is a causal pipeline, not merely adding noise to the current state.

### 5A.6 Truth vs nominal model; controlled mismatch

Start with exact-ZOH ideal unicycle integration under constant applied $(v,\omega)$:

$$
\begin{aligned}
x_{k+1} &= x_k + \frac{v_k}{\omega_k}\left[\sin(\theta_k+\omega_k\Delta t)-\sin\theta_k\right],\\
y_{k+1} &= y_k - \frac{v_k}{\omega_k}\left[\cos(\theta_k+\omega_k\Delta t)-\cos\theta_k\right],\\
\theta_{k+1} &= \operatorname{wrap}(\theta_k+\omega_k\Delta t).
\end{aligned}
$$

Use the straight-line limit as $|\omega|\to0$; implement a numerically stable small-$\omega$ branch. Disturbance modules can modify effective forward speed/yaw rate or inject bounded pose increments; record the exact convention and avoid mixing body-frame and world-frame noise.

For **learning residual dynamics**, deliberately distinguish `true_plant` from `nominal_model`. Example: true plant includes unknown actuator lag and speed-dependent slip; nominal MPC assumes ideal kinematics. Also evaluate a correctly specified classical baseline where practical, so gains are not merely due to withholding easily modeled physics from MPC.

### 5A.7 Configuration-driven scenarios (illustrative YAML)

```yaml
schema_version: 1
name: circle_lag_noise_seed42
seed: 42
simulation:
  dt_physics: 0.005
  dt_control: 0.05
  dt_sensor: 0.02
  duration_s: 30.0
reference:
  family: circle
  radius_m: 3.0
  speed_mps: 0.6
  direction: ccw
initial_error:
  dx_body_m: 0.3
  dy_body_m: -0.2
  dtheta_rad: 0.15
plant:
  integrator: exact_zoh
  actuator_tau_v_s: 0.12
  actuator_tau_omega_s: 0.08
  slip_fraction: 0.02
sensor:
  pose_std: [0.02, 0.02, 0.01]
  latency_s: 0.03
  dropout_probability: 0.0
command:
  latency_s: 0.05
  v_limits_mps: [0.0, 1.0]
  omega_limits_radps: [-1.5, 1.5]
  dv_max_mps2: 1.0
  domega_max_radps2: 3.0
benchmark:
  split: test
  suite_version: phase_a_v1
```

Validate configs before running: finite values, positive timesteps, feasible bounds, nonnegative latencies, legal probabilities, and reference feasibility under the chosen robot limits. Fail loudly for incompatible time grids; do not silently round. Use named scenarios plus overrides instead of controller-specific copies of scenario definitions.

### 5A.8 Fixed benchmark suite and experiment taxonomy

Use the **same scenarios and paired random seeds** for all controllers. Freeze `phase_a_v1` before tuning the final test set. Suggested tiers:

| Tier | Purpose | Representative cases | Metrics emphasized |
| --- | --- | --- | --- |
| B0 — Smoke | Infrastructure sanity | straight, circle, zero-noise | deterministic replay, finite values |
| B1 — Nominal | Core tracking | straight, circle, figure-eight, S-curve, stop/go | RMS/max position, heading, control |
| B2 — Recovery | Convergence basin | lateral offsets, heading offsets, wrong-way start | recovery time, failure rate |
| B3 — Robustness | Imperfect feedback | noise, bias, dropout, 20–100 ms sensor delay | degradation vs B1 |
| B4 — Dynamics | Model mismatch | actuator lag, speed-dependent slip, control delay | tracking, constraints, recovery |
| B5 — Stress | Operational limits | high curvature, near saturation, slow solver | deadline misses, solver failures |
| B6 — Geometry | Representation ablation | global translations/rotations of identical relative task | equivariance and error metrics |
| B7 — Learning OOD | Generalization | unseen trajectories, parameter ranges, disturbance families | failure rate, OOD performance |

**Evaluation protocol:** tune only on `train`/`dev` scenarios; evaluate once on frozen `test` suites; publish all attempted configurations or at least the tuning budget. Run e.g. 20 paired seeds per stochastic scenario initially and expand when failure-rate confidence intervals are wide. Deterministic scenarios need only one seed unless controller optimization itself is stochastic. Always report episode count, not just mean performance.

Define reference tracking errors from ground truth for scoring, while controller inputs come from estimated/delayed observations. Log both reference-time error and (where appropriate) nearest-path cross-track error, because path-following and timed tracking are not interchangeable. Record control constraint violations using **applied** and **commanded** controls separately. Declare failure thresholds and settling windows in the suite config before evaluation.

### 5A.9 Episode logging, replay, and synthetic dataset generation

Save one **episode manifest** (`JSON`) and one typed trajectory table (`Parquet` recommended, or compressed `NPZ` for minimal dependencies). Use one record per physics tick for truth/applied controls, and separate event tables for sensor samples, deliveries, controller decisions, and solver diagnostics; never fabricate missing observations by forward-filling without marking them.

**Minimum fields:** `episode_id`, `scenario_id`, `split`, `seed`, `sim_time`, `truth_pose`, `sensor_sample_time`, `observation_available_time`, `observation_pose`, `reference_id`, `reference_time`, `command_v/omega`, `applied_v/omega`, `solver_status`, `solve_wall_ms`, `disturbance_params`, `terminated_reason`, `config_hash`, `git_commit`.

Dataset views can be derived from immutable episode logs:

- **BC:** `(observation, reference_window) -> expert_command` with expert success flags and normalized features.
- **DAgger:** states visited by learner, relabeled by expert at the **same information state**; record query failures and the executed policy.
- **RL:** `(observation, action, reward, next_observation, terminated, truncated)` with action semantics (command vs applied) documented.
- **Dynamics learning:** `(state_or_belief, applied_control, dt) -> next_state_or_belief` with timestamps and observability caveats. Truth-state targets are acceptable for simulated system-ID research, but must be explicitly labeled as privileged supervision.

**Leakage prevention:** split by *trajectory family/parameters and episode IDs*, not by random rows from the same rollout. Keep normalization statistics fitted on train only. Preserve a genuinely held-out OOD suite. Expert demonstrations should cover recovery states, not only nominal perfect tracking. A policy trained on true state and evaluated on noisy delayed state is not a valid like-for-like comparison.

### 5A.10 Gymnasium wrapper and learning integration

The RL environment should wrap `EpisodeRunner`, with `reset(seed, options)` initializing a new scenario and `step(action)` advancing exactly **one controller interval** while physics and sensors run at their configured faster rates. The action is a **command**, not an instantaneous change in pose; observations reflect the sensor pipeline; reward uses ground truth only inside the environment, never exposed to the policy. Return `(obs, reward, terminated, truncated, info)` and keep reward weights versioned independently of benchmark metrics.

**Action scaling:** use bounded normalized actions, map them to physical commands, then pass through the *same* saturation/rate/delay/actuator pipeline as classical controllers. A separate wrapper can expose a `Box` observation vector for PPO/SAC while preserving the same underlying observation semantics. Include `check_env` tests and seed reproducibility tests.

### 5A.11 Runner pseudocode and reproducibility checks

```python
# Pseudocode: event scheduling and sensor delivery must be implemented explicitly.
def run_episode(scenario, controller, seed):
    runner = EpisodeRunner.from_config(scenario, seed=seed)
    controller.reset(seed=seed)
    while not runner.done:
        for event in runner.events_at_current_tick():
            if event.kind == "sensor_sample":
                runner.sample_sensor_and_schedule_delivery()
            elif event.kind == "sensor_delivery":
                runner.deliver_observation()
            elif event.kind == "control_tick":
                obs = runner.latest_delivered_observation()
                ref = runner.reference_window()
                result = controller.act(obs, ref)
                runner.enqueue_command(result)
            elif event.kind == "command_delivery":
                runner.make_command_eligible()
        runner.advance_actuator_and_physics_one_tick()
        runner.record_tick()
    return runner.finalize_episode()
```

**M0 acceptance tests (required before PID):**

1. Exact-ZOH matches straight-line/circular analytical solutions and a high-resolution RK4 oracle within declared tolerances.
2. Same config + seed + action sequence yields bitwise-identical or tolerance-identical traces on the same pinned environment; independent RNG streams prevent a new sensor draw from altering disturbance draws.
3. Zero-noise, zero-delay, zero-lag configuration reproduces the ideal baseline.
4. A 50 ms command delay causes no change in *applied* control before the scheduled delivery tick; a 30 ms sensor delay never exposes future truth.
5. Saturation and slew-rate bounds hold for commanded/eligible/applied values according to the documented order.
6. Sensor dropout creates a missing event and a stale observation with an explicit timestamp; it does not expose truth.
7. Metrics recomputed from saved logs match online summaries; replay with saved applied controls reproduces truth trajectory.
8. Two controllers on the same scenario share identical reference and exogenous noise realizations; compare against a scenario seed, not controller-specific random draws.
9. Dataset splits are disjoint by episode/trajectory identifiers, and no test normalization statistics enter training.
10. The Gymnasium adapter and offline runner agree for a fixed command sequence and seed.

### 5A.12 Build order and scope discipline

**M0a — Minimal deterministic engine:** exact-ZOH plant, reference generator, controller interface, runner, plot, truth logging. **M0b — Experiment platform:** frozen scenario catalog, evaluator, CSV/NPZ/JSON export, reproducible sweeps, replay. **M0c — Realism plugins:** independent seeded sensor noise, latency queues, actuator lag, command limits, disturbances, multi-rate scheduling. **M0d — Learning readiness:** dataset exporter, expert labeling contract, Gymnasium adapter, train/test split manifest. Finish M0a–M0c before PID; M0d can be completed before BC without changing the runner.

**Keep the implementation simple:** a single-process, vector-free reference runner is the source of truth. Parallel scenario execution and vectorized RL environments are optimization layers added only after correctness tests pass. If a feature requires modifying each controller, its abstraction boundary is probably wrong.

---

## 6. Mathematical and implementation tests

| Test | Expected property |
|---|---|
| Zero-input integration | Pose unchanged |
| Straight-line ZOH | Matches analytical straight-line solution |
| Constant-turn ZOH | Matches circular-arc solution |
| Angle wrapping | Correct near $\pm\pi$ |
| Jacobian finite difference | Analytic/autodiff Jacobian agrees with numerical derivative |
| Reference feasibility | Finite-difference motion consistent with reference $v_r,\omega_r$ |
| LQR Riccati | $P_k$ symmetric (within tolerance), finite gains |
| iLQR line search | Accepted steps decrease objective (within numerical tolerance) |
| MPC constraints | Feasible solutions respect configured bounds |
| Solver failure | Controlled fallback and explicit failure status |
| SE(2) Exp/Log | $\mathrm{Exp}(\mathrm{Log}(T))\approx T$ away from branch ambiguity |
| SE(2) composition | Inverse and composition identities |
| Rigid-frame equivariance | Equivalent transformed physical problem gives transformed trajectory |
| Determinism | Same seed/config gives same simulation outputs within tolerance |

For SE(2), test around zero rotation and near $\pm\pi$ separately. Compare Jacobians away from discontinuities. Keep angle error definitions consistent across all tests.

---

## 7. Milestones and acceptance criteria

### M0 — Infrastructure and mathematical baseline
- [ ] Unicycle exact-ZOH simulator, optional Euler test implementation
- [ ] Feasible straight/circle/figure-eight/S-curve references
- [ ] Path progress and timed-reference sampling
- [ ] Typed controller interface, config loader, deterministic multi-rate runner
- [ ] Separate truth / estimated observation / commanded / applied controls
- [ ] Seeded noise, sensor/control latency, actuator dynamics, disturbance plugins
- [ ] Immutable benchmark suite, episode replay, dataset export with split metadata
- [ ] Metrics, CSV/JSON logging, path/error/control plots
- [ ] Unit tests for integration, wrapping, and reference consistency

**Exit:** A dummy/feedforward controller can run end-to-end with reproducible outputs.

### M1 — PID + Pure Pursuit
- [ ] Heading/speed PID, anti-windup, filtered derivative
- [ ] Pure Pursuit lookahead with arclength progress
- [ ] Fixed/adaptive lookahead and gain sweeps
- [ ] Path-following benchmark; diagnose crossing/corner-cutting failures

**Exit:** Both controllers track basic feasible paths; comparison includes tuning sensitivity.

### M2 — LQR
- [ ] Derive time-varying linearization
- [ ] Verify Jacobians numerically
- [ ] Implement backward Riccati recursion and feedforward tracking
- [ ] Test small/large errors and stationary-reference caveat

**Exit:** Reproducible LQR trajectory tracking with documented region-of-validity tests.

### M3 — iLQR
- [ ] Rollout, cost derivatives, backward/forward passes
- [ ] Regularization, line search, convergence logging
- [ ] Nominal optimization and feedback policy execution
- [ ] Compare warm starts and optimization time separately

**Exit:** Cost decreases reliably on representative unconstrained tasks; disturbed feedback rollouts tested.

### M4 — Linear MPC
- [ ] Time-varying affine model and sparse QP
- [ ] Input/rate bounds and solver status logging
- [ ] Warm start, infeasibility policy, runtime percentiles

**Exit:** Constraints and deadline-miss statistics are reproducible.

### M5 — Nonlinear MPC — **core project completion**
- [ ] CasADi multiple-shooting NLP with matching dynamics/cost
- [ ] Bounds, warm start, solver failure handling
- [ ] Linear vs nonlinear prediction-error and runtime comparison
- [ ] Baseline report across all core controllers

**Exit:** Five major families (PID, Pure Pursuit, LQR, iLQR, MPC in linear/nonlinear forms) benchmarked with defensible caveats.

### M6 — Lie-group LQR and MPC — Phase A completion
- [ ] SE(2) Exp/Log/adjoint math tests
- [ ] Explicit left/right invariant error convention
- [ ] Lie-group LQR and Lie-linearized MPC first
- [ ] Rigid-transform and error-coordinate ablations
- [ ] Optional nonlinear group-valued MPC

**Exit:** Report demonstrates what changes with geometric representation and what does not.

### M7 — Phase A final benchmark and presentation
- [ ] Frozen test scenarios and tuning policy
- [ ] Multiple-seed robustness study
- [ ] Mean/P95/max timing + deadline misses
- [ ] Failure categories and confidence intervals
- [ ] Clear comparison plots and narrated animations
- [ ] README: assumptions, reproduction commands, findings, limitations

---

## 8. Recommended study order and conceptual checkpoints

1. **Feedback foundations:** body-frame error dynamics; PID anti-windup; path geometry and Pure Pursuit.
2. **Linear optimal control:** discrete linearization; controllability; quadratic costs; finite-horizon Riccati recursion.
3. **Nonlinear optimal control:** chain rule and derivatives; dynamic programming; iLQR vs DDP; line search and regularization.
4. **Numerical optimization:** convex QP vs nonconvex NLP; KKT conditions; sparse solvers; warm starts; infeasibility.
5. **Predictive control:** receding horizon; feedback vs planned open-loop controls; terminal costs and practical stability limitations.
6. **Geometric control:** Lie groups, algebra, Exp/Log, adjoint, invariant errors, coordinate-frame equivariance.
7. **Experimental methodology:** controls/ablations, fair tuning, timing methodology, statistical uncertainty, failure analysis.

**Questions you should be able to answer at the end:**

- Why does a heading-only PID fail to correct certain lateral errors?
- Why can a longer Pure Pursuit lookahead reduce oscillation but increase corner cutting?
- Why does a stationary unicycle linearization lose lateral controllability?
- What does the iLQR backward pass compute, and how does it differ from full DDP?
- Can iLQR serve as an MPC optimizer? What changes in a receding-horizon implementation?
- When is Linear MPC a convex QP, and when is a linearization affine rather than homogeneous?
- Why is NMPC not guaranteed to be globally optimal or real-time safe?
- Why does $SE(2)$ logarithmic pose error differ from simple body-frame translation error?
- How would you prove a claimed Lie-group improvement is due to geometry rather than different tuning/integration?
- What do P95 latency, deadline misses, and solver failure rate tell you that mean latency does not?

---

## 9. Learning-based trajectory tracking extension (Project 1B)

> **Status:** Phase B begins **after M6 and M7**, once all eight analytical controllers—including Lie-group LQR and Lie-group MPC—have been implemented and evaluated. Learning methods are additional experimental baselines, not replacements for the eight. Keep Project 2 (planning and obstacle avoidance) separate.

### 9.1 Why this is a good first robot-learning problem

A 2D differential-drive/unicycle tracking task has low-dimensional states and actions, cheap deterministic rollouts, an available optimal-control expert, and measurable closed-loop performance. A small multilayer perceptron (MLP) is a credible **first baseline**, not an assertion that a learned controller will outperform NMPC. Training on a laptop CPU can be practical for BC and small environments; RL runtime depends on episode count, parallelization, and hyperparameters. No camera, large GPU, Transformer, or physics engine is required to start.

**Research questions:**

- **RQ-L1:** Can a compact neural policy imitate an NMPC expert *in closed loop*, not merely predict its actions on held-out samples?
- **RQ-L2:** Does DAgger improve recovery from states induced by learner errors compared with offline behavior cloning?
- **RQ-L3:** Can PPO or SAC learn tracking from rewards, and how many simulator interactions does it take relative to collecting expert demonstrations?
- **RQ-L4:** Can learned residual actions or learned residual dynamics compensate for deliberately introduced actuator lag, slip, or model mismatch?
- **RQ-L5:** Which policies generalize to unseen path families, speeds, disturbances, and initial offsets?

### 9.2 Keep a single observation/action contract

The physical plant is the same as in Section 1. Distinguish the simulator's **true state** from the **observation** available to a policy. A minimal observation can be expressed as:

$$
\mathbf{o}_t = [e_x,e_y,\sin e_\theta,\cos e_\theta,
 v_r,\omega_r,v_{t-1},\omega_{t-1}]^\top.
$$

For anticipation, append a fixed-size horizon of future reference poses and velocities, expressed in the **current robot body frame**. Specify whether the policy is given current measured velocities (important with actuator lag). Avoid raw global $x,y$ unless testing frame dependence explicitly. Normalize features using **training-only** statistics; save normalization parameters with each model.

Use a continuous two-dimensional action, $\mathbf a_t=[a_v,a_\omega]^\top\in[-1,1]^2$, mapped to the plant's valid velocity-command ranges. For a forward-only robot, map $a_v$ into $[0,v_{\max}]$; map $a_\omega$ into $[-\omega_{\max},\omega_{\max}]$. Apply the **same actuator safety layer** as Project 1A. Record both requested and applied commands; action clipping is not proof of closed-loop stability.

### 9.3 Synthetic dataset generation and split design

An expert (preferably a validated NMPC controller) produces labeled demonstrations:

$$
\mathcal D_E=\{(\mathbf o_t,\mathbf u_t^*,\mathbf o_{t+1},
 \mathrm{episode\_id},\mathrm{scenario\_id})\}.
$$

Generate full episodes with randomized initial pose, reference curvature/speed, and optional sensor noise/actuator lag. Include both expert rollouts and **recovery states** (perturb initial states or states during rollouts, then query the expert). Only accept expert labels when its optimization succeeded; log failed expert queries instead of silently training on fallback commands.

**Split by entire trajectory families / scenario seeds, never by adjacent timesteps.** Otherwise temporally correlated samples leak into validation/test. Maintain three sets: (1) training, (2) validation for early stopping and tuning, (3) frozen test with held-out trajectories and disturbances. Include an in-distribution (ID) and out-of-distribution (OOD) test. Example: train on circles with radius 2–5 m; test circles of radius 6–8 m **and** a separate unseen S-curve family. The latter tests a different type of shift; neither is a guarantee of real-world generalization.

Start with modest dataset sizes (for example, tens to hundreds of thousands of labeled timesteps); increase only after plotting learning curves. Store expert version, solver tolerance, controller horizon, reference generator version, random seed, and model parameters in metadata.

### 9.4 L1 — Behavior cloning (supervised learning)

Train an MLP policy $\pi_\phi(\mathbf o_t)$ to predict expert actions:

$$
\mathcal L_{\mathrm{BC}}(\phi)
=\frac{1}{N}\sum_{i=1}^{N}
\left(\pi_\phi(\mathbf o_i)-\mathbf u_i^*\right)^\top
W\left(\pi_\phi(\mathbf o_i)-\mathbf u_i^*\right).
$$

Use a **2-layer MLP with 64–128 hidden units per layer** as the first architecture, Adam, standardized inputs, and early stopping. Scale output targets/actions consistently; the two action dimensions have different physical units. Report offline action MSE **and** the complete closed-loop tracking benchmark. Low offline MSE is not sufficient: small action errors can accumulate and shift the robot into unseen states (covariate shift).

```python
import torch
from torch import nn

class TrackingMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Normalized commands in [-1, 1]; map to physical bounds outside.
        return self.net(obs)
```

**Ablations:** observation without/with reference preview; 32/64/128 hidden units; training set size; expert type (LQR vs NMPC); training with/without recovery-state labels. Compare inference latency with NMPC *solve latency*, while separately accounting for expert data-generation and training cost.

### 9.5 L2 — DAgger (interactive imitation learning)

DAgger addresses distribution shift by collecting observations from **the learner's own rollouts** and querying the expert at those states:

$$
\mathcal D_{j+1}=\mathcal D_j\cup
\{(\mathbf o,\pi_E(\mathbf o)):\mathbf o\sim d_{\pi_j}\}.
$$

Here $d_{\pi_j}$ denotes the state/observation visitation distribution of the current learned policy. In each iteration: train on aggregated data, roll out the learner on **training scenarios**, ask the expert for labels at visited states, append labels, retrain, and evaluate on the frozen test set only at designated checkpoints. A safer initial implementation can mix expert and learner actions during data collection; record the mixing schedule.

**Primary comparison:** BC vs DAgger on large initial offsets, novel curves, and recovery after injected perturbations. Plot failure rate against **number of expert queries**, not just epochs; DAgger uses additional supervision. Expert infeasibility and unsafe learner rollouts need explicit fallback/termination rules.

### 9.6 L3 — Model-free reinforcement learning (PPO, then SAC)

Wrap the simulator in a Gymnasium-style interface. State exactly what the agent observes, which action is applied, how long one step lasts, and what ends an episode. RL does not require expert labels; it learns from reward and interaction.

$$
J(\phi)=\mathbb E_{\pi_\phi}\left[
\sum_{t=0}^{T-1}\gamma^t r_t\right].
$$

A starting reward (weights must be tuned and reported) is:

$$
r_t=-w_p\|\mathbf e_{p,t}\|^2
-w_\theta\,\mathrm{wrap}(e_{\theta,t})^2
-w_u\|\mathbf u_t\|^2
-w_{\Delta u}\|\mathbf u_t-\mathbf u_{t-1}\|^2.
$$

Use a clearly defined success bonus, failure penalty, and time limit if appropriate. Check for reward loopholes: an agent may stop moving if effort penalties outweigh tracking reward; a path follower may cut corners if progress is rewarded too aggressively. For time-indexed trajectories, penalize **time-aligned** errors rather than only nearest-path distance.

- **PPO:** on-policy actor-critic; straightforward first training loop, but often requires more environment samples.
- **SAC:** off-policy actor-critic with replay buffer and entropy regularization; useful for continuous actions and potentially better sample efficiency, but more components to debug.

Start with a trusted implementation (e.g., Stable-Baselines3) before attempting PPO/SAC from scratch. Compare **environment steps, wall-clock training time, random-seed variability, closed-loop success, and test latency**. Evaluate stochastic policies with a declared deterministic/stochastic action setting.

```python
class TrackingEnv:
    def reset(self, *, seed=None, options=None):
        # Seed scenario RNG; sample training trajectory and initial state.
        # Return observation and info.
        raise NotImplementedError

    def step(self, action):
        # Scale action, apply actuator bounds, integrate plant,
        # sample next reference, calculate reward and termination.
        # Return (observation, reward, terminated, truncated, info).
        raise NotImplementedError
```

### 9.7 L4 — Hybrid approaches: residual RL and learned dynamics MPC

**Residual RL:** let a baseline controller handle nominal tracking, while a policy learns a bounded correction:

$$
\mathbf u_t=\operatorname{clip}_{\mathcal U}
\left(\mathbf u_t^{\mathrm{base}}+\alpha\,\pi_\phi(\mathbf o_t)\right).
$$

Start with LQR + residual PPO/SAC. Vary $\alpha$ and mismatch strength; compare with the unmodified LQR and a pure RL policy. A residual controller can still destabilize the system; do not imply formal safety without additional analysis.

**Learned residual dynamics MPC:** deliberately separate the simulator's richer plant from the controller's nominal model:

$$
\mathbf x_{t+1}=f_{\mathrm{nom}}(\mathbf x_t,\mathbf u_t)
+\hat{\mathbf d}_\phi(\mathbf x_t,\mathbf u_t,\mathbf h_t).
$$

The residual can model actuator lag, slip, or history-dependent effects; if these make the physical process partially observed, add measured velocity/history or augment the state. Compare nominal MPC, residual-model MPC, and an **oracle-model** MPC that is explicitly labeled as having privileged true-dynamics knowledge. A learned model provides little reason for improvement if the simulator and nominal controller already use identical perfect dynamics. Measure both one-step prediction error and **multi-step rollout / closed-loop tracking** error; one-step accuracy alone can be misleading.

### 9.8 Shared learning-based benchmark protocol

Keep all six original control metrics from Section 4: RMS/max position error; heading error; effort/smoothness; applicable settling time; mean/P95/max per-step computation time; constraint violations/failure rate. Add:

| Metric | Why it matters |
| --- | --- |
| Offline action MAE/MSE | Imitation fit; **not** a substitute for closed-loop evaluation |
| Closed-loop success and recovery rate | Compounding error and recovery from perturbations |
| Expert queries / labeled timesteps | Imitation-data efficiency and DAgger cost |
| Environment interaction steps | RL sample efficiency |
| Training wall-clock and inference P95 | Offline cost vs online control feasibility |
| Performance across 5+ seeds (when feasible) | RL training variance and uncertainty |
| ID vs OOD performance gap | Generalization to unseen references/conditions |
| Reward vs physical tracking metrics | Detect reward hacking and metric mismatch |
| Model prediction vs rollout error | Validate learned-dynamics utility |

**Fairness rules:** same plant, time step, reference information horizon, actuator limits, observation noise, and test episodes. Explicitly disclose privileged information, such as an expert's true dynamics or an MPC's longer preview. Do **not** equate RL training steps with expert labels; report them separately. For timing, distinguish data collection, offline training, online inference, and online solver computation. Preserve the original no-obstacle boundary.

**Experiment grid:**

1. **ID tracking:** straight, circle, figure-eight, S-curve, start/stop.
2. **Initial-error recovery:** lateral offsets and heading perturbations, including outside expert-training distribution.
3. **OOD trajectories:** held-out radii, curvature patterns, speed profiles, and scenario seeds.
4. **Disturbance robustness:** sensor noise, actuator lag, bounded slip, model mismatch.
5. **Data scaling:** BC performance vs labeled dataset size; DAgger vs expert-query budget.
6. **RL learning curves:** PPO/SAC return **and** tracking success vs environment steps.
7. **Ablations:** no reference preview; no action history; no domain randomization; no residual correction.

### 9.9 Suggested module structure and experiment configs

```text
src/control_benchmark/
  learning/
    datasets.py            # episode-based expert demonstrations
    observations.py        # body-frame features + train-only normalization
    bc.py                  # PyTorch MLP and supervised training
    dagger.py              # learner rollouts + expert relabeling
    env.py                 # Gymnasium-compatible tracking environment
    rl.py                  # PPO / SAC wrappers and evaluation
    residual_policy.py     # baseline + bounded learned correction
    learned_dynamics.py    # supervised residual model
    evaluate.py            # same tracking benchmark adapter
configs/learning/
  bc.yaml
  dagger.yaml
  ppo.yaml
  sac.yaml
  residual_rl.yaml
  residual_mpc.yaml
```

Store `checkpoint`, feature normalization, observation schema, action mapping, training split, scenario generator version, seed, and package versions together. Ensure learned controllers implement the same `compute_control(...)` interface as analytical controllers (with an adapter if needed).

### 9.10 Learning milestones and acceptance criteria

- **L0 — Data and interfaces:** [ ] deterministic dataset generator; [ ] episode-level train/val/test split; [ ] observation/action tests; [ ] expert failure filtering. **Exit:** identical seeds reproduce labels and episode metadata.
- **L1 — BC:** [ ] small MLP; [ ] loss curves; [ ] offline MAE; [ ] closed-loop benchmark; [ ] latency and dataset-size ablation. **Exit:** demonstrate whether offline accuracy predicts closed-loop success.
- **L2 — DAgger:** [ ] learner rollout/relabeling loop; [ ] expert-query accounting; [ ] recovery tests. **Exit:** defensible BC-vs-DAgger comparison on frozen scenarios.
- **L3 — PPO/SAC:** [ ] Gymnasium environment; [ ] reward diagnostics; [ ] multiple training seeds; [ ] learning curves and OOD tests. **Exit:** reproducible RL results with sample and compute budgets.
- **L4 — Hybrid control (optional):** [ ] residual RL and/or learned residual dynamics; [ ] deliberate model mismatch; [ ] oracle comparison; [ ] failure analysis. **Exit:** quantify where learning helps and where it does not.

**Recommended first stopping point:** finish L0–L2 after M5. This delivers a coherent progression from optimal control to supervised imitation to interactive imitation **without waiting** for PPO/SAC or Lie-group MPC.

### 9.11 Conceptual checkpoints

- Why is trajectory tracking an MDP, and when does actuator lag make observations insufficient for the Markov property?
- Why does minimizing action MSE not guarantee low closed-loop tracking error?
- What does DAgger change about the distribution of supervised examples?
- How do PPO and SAC differ in on-policy/off-policy learning and data reuse?
- Why can a low-effort reward cause the robot to stop?
- When does learned residual dynamics improve on a known analytical model, and when is it redundant?
- What would count as OOD generalization rather than a random split of nearly identical timesteps?
- Why is a 2-layer MLP a strong baseline for this low-dimensional task, and what evidence would justify a larger architecture?

---

## 10. Final report template

```markdown
# Differential-Drive Control Benchmark: Results

## Research questions and hypotheses
## Robot model, references, and assumptions
## Controllers and tuning budgets
## Experimental protocol and held-out scenarios
## Nominal tracking: path vs timed trajectory
## Initial-error recovery and stability limitations
## Robustness: noise, lag, and model mismatch
## Constraint satisfaction and solver failures
## Runtime: mean, P95, max, and deadline misses
## Lie-group error representation ablations
## Discussion: why methods differ
## Threats to validity and limitations
## Reproducibility instructions
## Learning-based tracking: BC/DAgger/RL/hybrid results (if completed)
## Conclusions and next steps (Project 2)
```

## 11. Final scope decision

**Infrastructure must-have:** extensible deterministic simulator, timing/noise/latency pipeline, reproducible benchmark and dataset export. **Phase A goal:** implement and benchmark all eight classical/model-based controllers before Phase B.  
**Phase A (all eight):** PID, Pure Pursuit, LQR, iLQR, Linear MPC, NMPC, Lie-group LQR, Lie-group MPC. **Phase B (after Phase A):** BC → DAgger → PPO → SAC → Residual RL → Learning-based MPC; later two may be scoped as advanced experiments.  
**Out of scope:** obstacle maps, collision avoidance, global/local planning, multi-robot coordination, full rigid-body dynamics, ROS 2 integration.

The project is complete when the results explain **assumptions, mechanisms, trade-offs, and failure modes** with reproducible evidence—not merely when eight classes compile or eight lines appear on one plot.
