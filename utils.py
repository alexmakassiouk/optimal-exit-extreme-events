import math

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


def d_2_func(params: ModelParams) -> float:
    """
    Negative root of characteristic equation
    """
    term_1 = 0.5 - params.mu / params.sigma**2
    term_2 = math.sqrt(
        (params.mu / params.sigma**2 - 0.5) ** 2
        + 2 * (params.r + params.lambd) / params.sigma**2
    )
    return term_1 - term_2


# --- Absorbing State ---
def n_bar_func(params: ModelParams) -> int:
    """
    Calculates the absorbing jump level n_bar.
    When n >= n_bar, profit is always negative, and the firm stops.
    """
    return math.ceil(1 / params.phi)


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


# --- Stopping Threshold functions ---
def x_star_top_func(params: ModelParams) -> float:
    d_1 = d_1_func(params)
    n_bar = n_bar_func(params)
    return (
        ((d_1 - 1) / d_1)
        * (params.c + params.r * params.L)
        / (1 - params.phi * (n_bar - 1))
    )


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
