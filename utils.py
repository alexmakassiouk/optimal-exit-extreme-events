import math
from typing import Tuple

from .params import ModelParams


# --- Characteristic Roots ---
def d_1_func(params: ModelParams) -> float:
    """
    Positive root of characteristic equation
    """
    term_1 = 0.5 - params.mu / params.sigma**2
    term_2 = math.sqrt(
        (params.mu / params.sigma**2 - 0.5) ** 2
        + 2 * (params.r + params.lambd) / params.sigma**2
    )
    return term_1 + term_2


def d_2_func(params: ModelParams):
    """
    Negative root of characteristic equation
    """
    term_1 = 0.5 - params.mu / params.sigma**2
    term_2 = math.sqrt(
        (params.mu / params.sigma**2 - 0.5) ** 2
        + 2 * (params.r + params.lambd) / params.sigma**2
    )
    return term_1 - term_2


# --- Terms that reappear ---
def S_1_func(params: ModelParams):
    return params.lambd / (params.r + params.lambd)


def S_2_func(params: ModelParams):
    return params.lambd / (params.r + params.lambd - params.mu)


# --- Absorbing State ---
def n_bar_func(params: ModelParams) -> int:
    """
    Calculates the absorbing jump level n_bar.
    When n >= n_bar, profit is always negative, and the firm stops.
    """
    return math.ceil(1 / params.phi)


def phi_interval_for_n_bar(n_bar: int) -> Tuple[float, float]:
    """Return the interval [phi_lower, phi_upper) that maps to the given n_bar via ceil(1/phi)."""
    if n_bar < 1:
        raise ValueError("n_bar must be >= 1")
    if n_bar == 1:
        return (1.0, math.inf)

    phi_lower = 1.0 / n_bar
    phi_upper = 1.0 / (n_bar - 1)
    return (phi_lower, phi_upper)


# --- Perpetuity Value ---
def perpetuity_value(params: ModelParams, x: float, n: int) -> float:
    """
    Calculates V_hat(x,n) which is the expected PV of operating forever
    with n initial jumps.
    V_hat(x,n) = x/(r-mu) * (1-phi * n - phi*lambda/(r-mu)) - c/r
    such that V(x,n) = V_hat(x,n) + v(x,n)
    """
    if params.r == params.mu:
        # Avoid division by zero
        raise ZeroDivisionError
    term_1 = x / (params.r - params.mu)
    term_2 = 1 - params.phi * n - (params.phi * params.lambd) / (params.r - params.mu)
    term_3 = params.c / params.r

    return term_1 * term_2 - term_3


def perpetuity_value_prime(params: ModelParams, x: float, n: int) -> float:
    """
    Calculates the derivative of V_hat(x,n) with respect to x.
    V_hat'(x,n) = 1/(r-mu) * (1 - phi*n - phi*lambda/(r-mu))
    """
    if params.r == params.mu:
        raise ZeroDivisionError

    term_1 = 1 / (params.r - params.mu)
    term_2 = 1 - params.phi * n - (params.phi * params.lambd) / (params.r - params.mu)

    return term_1 * term_2


# --- Payoff and Particular Solution Functions ---
def g_func(params: ModelParams, x: float, n: int, k: int) -> float:
    # Payoff for price level x, start level of jumps n, and number of additional jumps k
    # Setting k=0 gives the payoff for exercising right now
    term_1 = (params.lambd / (params.r + params.lambd)) ** k * (
        params.c / params.r + params.L
    )
    term_2 = (
        (params.lambd / (params.r + params.lambd - params.mu)) ** k
        * x
        / (params.r - params.mu)
        * (
            1
            - params.phi * (n + k)
            - params.phi * params.lambd / (params.r - params.mu)
        )
    )
    return term_1 - term_2


def g_prime_func(params: ModelParams, x: float, n: int, k: int) -> float:
    """
    Derivative of g_func with respect to x.
    Needed for smooth-pasting.
    """
    if k == 0:
        # Value of stopping is g(x,i,0) = L - V_hat(x,i)
        # Derivative is g'(x,i,0) = -V_hat'(x,i)
        return -perpetuity_value_prime(params, x, n)

    # Derivative of the particular solution
    term_2_prime = (
        (params.lambd / (params.r + params.lambd - params.mu)) ** k
        * 1
        / (params.r - params.mu)
        * (
            1
            - params.phi * (n + k)
            - params.phi * params.lambd / (params.r - params.mu)
        )
    )
    return -term_2_prime


# --- Stopping Threshold functions ---
def x_star_top_func(params: ModelParams):
    d_1 = d_1_func(params)
    n_bar = n_bar_func(params)
    return (
        ((d_1 - 1) / d_1)
        * (params.c + params.r * params.L)
        / (1 - params.phi * (n_bar - 1))
    )


def x_star(params: ModelParams, n: int, k: int, d: float):
    # Needs to be implemented based on recursive formulas
    if n == n_bar_func(params) - 1:
        return x_star_top_func(params)
    raise NotImplementedError


def zeta_func(params: ModelParams, n: int, k: int, d: float) -> float:
    # This function will need the solved A, B coefficients and the x_stars
    term_1 = (
        d
        * (params.lambd / (params.r + params.lambd)) ** (k - 1)
        * ((params.c + params.r * params.L) / (params.r + params.lambd))
    )
    term_2 = (
        (1 - d)
        * ((params.lambd) / (params.r + params.lambd - params.mu)) ** (k - 1)
        * x_star(params, n, k, d)
        / (params.r + params.lambd - params.mu)
        * (1 - params.phi * (n + k - 1))
    )
    return term_1 + term_2


def zeta_func_alt(params: ModelParams, n: int, k: int, d: float, x_npkm1) -> float:
    # This function will need the solved A, B coefficients and the x_stars
    term_1 = (
        d
        * (params.lambd / (params.r + params.lambd)) ** (k - 1)
        * ((params.c + params.r * params.L) / (params.r + params.lambd))
    )
    term_2 = (
        (1 - d)
        * ((params.lambd) / (params.r + params.lambd - params.mu)) ** (k - 1)
        * x_npkm1
        / (params.r + params.lambd - params.mu)
        * (1 - params.phi * (n + k - 1))
    )
    return term_1 + term_2


# --- New formulas testing
def K_func(params: ModelParams):
    term_1 = (params.c / params.r) + params.L
    term_2 = 1 - params.lambd / (params.lambd + params.r)
    return term_1 * term_2


def D_func(params: ModelParams):
    term_1 = 1 - 2 * params.phi - params.phi * params.lambd / (params.r - params.mu)
    term_2 = params.lambd / (params.r + params.lambd - params.mu)
    term_3 = 1 - 3 * params.phi - params.phi * params.lambd / (params.r - params.mu)
    return term_1 - term_2 * term_3


def x_star_top_func_alt(params: ModelParams):
    d_2 = d_2_func(params)
    K = K_func(params)
    D = D_func(params)
    return K * d_2 * (params.r - params.mu) / (D * (d_2 - 1))


import numpy as np
import matplotlib.pyplot as plt


def main():
    params = ModelParams(r=0.05, mu=0.03, sigma=0.1, lambd=0.01, phi=0.34, c=0.1, L=1.0)
    n = 2
    x_vals = np.linspace(0.1, 50, 10000)
    y_vals = [perpetuity_value(params, x, n) for x in x_vals]

    plt.figure(figsize=(8, 4))
    label = "\n".join(
        [
            rf"$\hat{{V}}(x, n={n})$",
            rf"$r = {params.r}$",
            rf"$\mu = {params.mu}$",
            rf"$\sigma = {params.sigma}$",
            rf"$\lambda = {params.lambd}$",
            rf"$\phi = {params.phi}$",
            rf"$c = {params.c}$",
            rf"$L = {params.L}$",
        ]
    )
    plt.plot(x_vals, y_vals, label=label)
    plt.axhline(0, color="gray", linewidth=0.7)
    plt.xlabel("x")
    plt.ylabel("perpetuity_value")
    plt.legend()
    plt.show()


# main()


def solve_equation_for_x_star(func, a, b, tol=1e-6, max_iter=100):
    """
    Finds the root of the equation func(x) = 0 within the interval [a, b]
    using the bisection method.
    """
    fa = func(a)
    fb = func(b)

    if fa * fb >= 0:
        print("Bisection method may fail if f(a) and f(b) have the same sign.")
        # Potentially raise an error or handle as needed
        # For now, returning the midpoint as a guess.
        return (a + b) / 2

    for _ in range(max_iter):
        c = (a + b) / 2
        fc = func(c)

        if abs(fc) < tol or (b - a) / 2 < tol:
            return c

        if fa * fc < 0:
            b = c
            # fb = fc # No need to re-evaluate fb
        else:
            a = c
            fa = fc

    return (a + b) / 2
