"""Shared analytical core for the optimal-stopping-with-extreme-events model.

This package contains the base exit-only analytical solver and its building
blocks. It is intended to be extractable as a standalone package if published.
"""

from .params import ModelParams
from .utils import (
    d_1_func,
    d_2_func,
    n_bar_func,
    perpetuity_value,
    g_func,
    x_star_top_func,
    zeta_func_alt,
)
from .main_solver import (
    LevelSolution,
    solve_all_levels,
    solve_conventional_model,
    V_hat_conventional,
)

__all__ = [
    "ModelParams",
    "d_1_func",
    "d_2_func",
    "n_bar_func",
    "perpetuity_value",
    "g_func",
    "x_star_top_func",
    "zeta_func_alt",
    "LevelSolution",
    "solve_all_levels",
    "solve_conventional_model",
    "V_hat_conventional",
]
