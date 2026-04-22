"""Shared analytical core for the optimal-stopping-with-extreme-events model.

This package contains the base exit-only analytical solver and its building
blocks. It is intended to be extractable as a standalone package if published.
"""

from .params import ModelParams
from .utils import (
    d_1_func,
    d_2_func,
    S_1_func,
    S_2_func,
    n_bar_func,
    phi_interval_for_n_bar,
    perpetuity_value,
    perpetuity_value_prime,
    g_func,
    g_prime_func,
    x_star_top_func,
    x_star,
    zeta_func,
    zeta_func_alt,
    K_func,
    D_func,
    x_star_top_func_alt,
    solve_equation_for_x_star,
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
    "S_1_func",
    "S_2_func",
    "n_bar_func",
    "phi_interval_for_n_bar",
    "perpetuity_value",
    "perpetuity_value_prime",
    "g_func",
    "g_prime_func",
    "x_star_top_func",
    "x_star",
    "zeta_func",
    "zeta_func_alt",
    "K_func",
    "D_func",
    "x_star_top_func_alt",
    "solve_equation_for_x_star",
    "LevelSolution",
    "solve_all_levels",
    "solve_conventional_model",
    "V_hat_conventional",
]
