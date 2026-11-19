# Contributing

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --group test
pytest
```

## Adding a controller

1. Create `diffdrive/controllers/<name>.py` with a `Controller` subclass,
   a frozen `Config` dataclass, `reference_request()`, and `compute()`.
   Decorate it with `@register("<name>")` and import the module in
   `diffdrive/controllers/__init__.py`.
2. Keep it pure: use only `obs`, `ref` and `self.ctx`. Never read simulator state.
3. Return `Status.FALLBACK` (with a safe command) when a solver fails. Do not
   hide failures.
4. Add unit tests for the maths (Jacobians, Riccati symmetry, constraint
   satisfaction). The shared acceptance tests in
   `tests/test_controller_acceptance.py` run automatically once
   `implemented = True`.
5. Add the controller to `configs/suites/nominal.yaml` and include benchmark
   results in the PR description.

A controller doesn't have to live in this repo: pass `module:Class` or
`path/to/file.py:Class` to `--controller` (or in a suite YAML).

## Adding a map or scenario

* Maps: `python maps/generate.py ...` writes `maps/yaml/mapNN.yaml` + `maps/csv/mapNN.csv` and
  refreshes the catalogue. Published maps are benchmark fixtures: never edit
  one, add a new number. Any tool may write maps (columns `s,x,y,yaw,kappa`).
* Scenarios: a YAML file in `configs/scenarios/` describing conditions and a
  default `map`. Its reference must pass `check_feasibility` under the
  scenario's command limits (`tests/test_sim.py` checks every scenario).

## Conventions

* SI units; world frame x-forward / y-left; heading CCW; wrap angle *differences*.
* Docstrings state assumptions and equations. Cite the spec section.
* Determinism: draw randomness only from the provided RNGs.
