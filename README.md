# optimal-exit-extreme-events

Analytical solver for an optimal exit problem with extreme-event shocks.

This repository contains the cleaned core implementation used in our thesis work,
intended to serve as a compact and reproducible appendix artifact.

## Scope

- Closed-form and recursive building blocks for the exit model.
- Multi-level threshold solver for the jump-augmented setting.
- Conventional benchmark solver (no jump risk).
- Minimal package API through [__init__.py](__init__.py).

Plotting and animation utilities have been intentionally removed so the repository
remains focused on the analytical core.

## Repository Layout

- [params.py](params.py): immutable model parameter container and admissibility checks.
- [utils.py](utils.py): mathematical primitives used by the solver.
- [main_solver.py](main_solver.py): recursive threshold/value-function solvers.
- [__init__.py](__init__.py): package-level exports.

## Quick Use

```python
from optimal_exit_extreme_events import ModelParams, solve_all_levels

params = ModelParams(
	r=0.05,
	mu=0.02,
	sigma=0.1,
	lambd=0.1,
	phi=0.2,
	c=2.0,
	L=-10.0,
)

solutions = solve_all_levels(params)
x0_star = solutions[0].x_star
```

## Notes

- The solver assumes a valid parameter region, enforced in [params.py](params.py).
- Numerical root-finding uses `scipy.optimize.newton`.
