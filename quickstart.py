"""Quick-start for the analytical solver core.

Run from repository root:
    python quickstart.py

This script demonstrates package loading and validates a few core invariants:
- solver runs end-to-end,
- expected number of states is returned,
- outputs are finite,
- benchmark (no-jump) solver executes.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


def _load_repo_as_package():
    """Load repository root as a package so relative imports work.

    The project currently exposes package modules at repo root (with `__init__.py`).
    This helper allows quick-start execution without requiring installation.
    """

    repo_root = Path(__file__).resolve().parent
    package_name = "optimal_exit_extreme_events"
    init_file = repo_root / "__init__.py"

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to build import spec for repository package.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    pkg = _load_repo_as_package()

    params = pkg.ModelParams(
        r=0.05,
        mu=0.02,
        sigma=0.1,
        lambd=0.1,
        phi=0.2,
        c=2.0,
        L=-10.0,
    )

    solutions = pkg.solve_all_levels(params)
    n_bar = math.ceil(1.0 / params.phi)

    # --- Smoke checks ---
    assert len(solutions) == n_bar, "Unexpected number of solved levels."

    x_stars = [solutions[n].x_star for n in range(n_bar)]
    assert all(math.isfinite(x) for x in x_stars), "Non-finite threshold encountered."

    v0 = solutions[0].v(params, x_stars[0])
    assert math.isfinite(v0), "Non-finite value at first threshold."

    conventional = pkg.solve_conventional_model(params)
    assert math.isfinite(conventional.x_star), "Conventional threshold is non-finite."

    print("Smoke test passed.")
    print(f"n_bar: {n_bar}")
    print(f"x_0^*: {x_stars[0]:.6f}")
    print(f"x_1^*: {x_stars[1]:.6f}")
    print(f"x_2^*: {x_stars[2]:.6f}")
    print(f"x_3^*: {x_stars[3]:.6f}")
    print(f"x_(n_bar-1)^*: {x_stars[-1]:.6f}")
    print(f"Conventional x^*: {conventional.x_star:.6f}")


if __name__ == "__main__":
    main()
