import math
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple, overload

from scipy.optimize import newton

from .params import ModelParams
from .utils import (
    n_bar_func,
    d_1_func,
    d_2_func,
    g_func,
    x_star_top_func,
    zeta_func_alt,
)


CoeffKey = Tuple[int, int, int]
CoeffDict = Dict[CoeffKey, float]


@dataclass
class LevelSolution:
    """Container for one level n.

    - x_star: optimal stopping threshold x_n^*
    - v: continuation value function v(x,n)
    """

    x_star: float
    v: Callable[[ModelParams, float], float]


def _zeta(params: ModelParams, n: int, k: int, d: float, x_star_nkp1: float) -> float:
    """Wrapper for the forcing term using x_{n+k-1}^* explicitly."""

    return zeta_func_alt(params, n, k, d, x_star_nkp1)


@overload
def solve_all_levels(params: ModelParams, n_bar: Optional[int] = ..., debug_n: None = ..., return_coeffs: Literal[False] = ...) -> Dict[int, LevelSolution]: ...
@overload
def solve_all_levels(params: ModelParams, n_bar: Optional[int] = ..., *, debug_n: int, return_coeffs: Literal[False] = ...) -> Callable[..., float]: ...
@overload
def solve_all_levels(params: ModelParams, n_bar: Optional[int] = ..., debug_n: None = ..., *, return_coeffs: Literal[True]) -> Tuple[Dict[int, LevelSolution], CoeffDict, CoeffDict]: ...

def solve_all_levels(
    params: ModelParams,
    n_bar: Optional[int] = None,
    debug_n: Optional[int] = None,
    return_coeffs: bool = False,
):
    """Solve thresholds x_n^* and value functions v(x,n) for all n=0,...,n_bar-1.

    This is a generalisation of `n_bar_3/main_solver_rita.py` to arbitrary n_bar,
    following the analytical formulas in the LaTeX description.
    
    If debug_n is provided, returns the implicit function F_n(x) for that level.
    If return_coeffs is True, returns (solutions, A, B).
    """

    if n_bar is None:
        n_bar = n_bar_func(params)

    d1 = d_1_func(params)
    d2 = d_2_func(params)

    # --- Thresholds x_n^* ---
    x_star: Dict[int, float] = {}

    # Top threshold x_{n_bar-1}^* via explicit formula (10)
    n_top = n_bar - 1
    x_star[n_top] = x_star_top_func(params)
    # x_star_test = x_star_top_func(params)

    # --- Coefficients A_{n,k,j}, B_{n,k,j} ---
    # Use nested dicts keyed by (n,k,j)
    A: CoeffDict = {}
    B: CoeffDict = {}

    # Growth condition: for all n and j, A_{n, k_max, j} = 0
    # where k_max(n) = n_bar - n
    for n in range(0, n_bar):
        k_max = n_bar - n
        for j in range(0, k_max):
            A[(n, k_max, j)] = 0.0

    # Base coefficients for k = 1 using zeta(n,1,d)
    # A_{n,1,0} for n=0,...,n_bar-2; B_{n,1,0} for n=0,...,n_bar-1
    # We do not yet know x_n^* for n < n_top; these will be filled after
    # thresholds are solved. For now we only set B for the top level.
    x_star_n_top = x_star[n_top]
    zeta_n1_d1 = _zeta(params, n_top, 1, d1, x_star_n_top)
    B[(n_top, 1, 0)] = (x_star_n_top ** (-d2)) / (d1 - d2) * zeta_n1_d1
    # A_{n_top,1,0} = 0 by growth condition of base case
    A[(n_top, 1, 0)] = 0.0

    # Helper to compute B_{n,1,0} and A_{n,1,0} once x_n^* is known for n < n_top
    def compute_A1B1_for_level(n: int) -> None:
        if n == n_top:
            return
        xn = x_star[n]
        zeta_n1_d2 = _zeta(params, n, 1, d2, xn)
        zeta_n1_d1 = _zeta(params, n, 1, d1, xn)
        A[(n, 1, 0)] = (xn ** (-d1)) / (d2 - d1) * zeta_n1_d2
        B[(n, 1, 0)] = (xn ** (-d2)) / (d1 - d2) * zeta_n1_d1

    # --- Undetermined coefficients (log terms) via (8) and (9), per level ---
    def compute_log_coeffs_for_level(n: int) -> None:
        """Fill A_{n,k,j}, B_{n,k,j} for fixed n, all k>=2, j>=1.

        This is called only after all required higher-level coefficients
        (n+1, . , .) are known, so dependency ordering is respected.
        """

        k_max = n_bar - n

        # A-coeffs: k = 2..k_max-1
        for k in range(2, k_max):
            for j in range(1, k):
                s = 0.0
                for ell in range(j - 1, k - 1):
                    num = ((-1) ** (ell + 1 - j)) * math.factorial(ell)
                    den = math.factorial(j) * (d1 - d2) ** (ell + 2 - j)
                    s += num / den * A.get((n + 1, k - 1, ell), 0.0)
                A[(n, k, j)] = -(2 * params.lambd / (params.sigma**2)) * s

        # B-coeffs: k = 2..k_max
        for k in range(2, k_max + 1):
            for j in range(1, k):
                s = 0.0
                for ell in range(j - 1, k - 1):
                    num = ((-1) ** (ell + 1 - j)) * math.factorial(ell)
                    den = math.factorial(j) * (d2 - d1) ** (ell + 2 - j)
                    s += num / den * B.get((n + 1, k - 1, ell), 0.0)
                B[(n, k, j)] = -(2 * params.lambd / (params.sigma**2)) * s

    # --- Threshold recursion for n < n_bar-1 ---
    def F_for_level(n: int, x_n: float, verbose: bool = False) -> float:
        """Implicit equation F_n(x_n) = 0 defining x_n^*.

        This generalises F_x0_star and F_x1_star patterns.
        """

        d1 = d_1_func(params)
        d2 = d_2_func(params)

        # Temporarily set x_n and recompute A_{n,1,0}, B_{n,1,0}
        x_star[n] = x_n
        compute_A1B1_for_level(n)

        k_max = n_bar - n

        # For k = 2,...,k_max-1 we need A_{n,k,0}, B_{n,k,0} via (7) and (6)
        # using the already-computed log-coefficients (assumed filled).

        # Compute A_{n,k,0} for k=2,...,k_max-1 (if any)
        for k in range(2, k_max):
            x_glue = x_star[n + k - 1]
            ln_x = math.log(x_glue)

            if k != 2:
                sum_term = 0.0
                for j in range(1, k - 1):
                    Bij = B.get((n, k, j), 0.0) - B.get((n, k - 1, j), 0.0)
                    Aij = A.get((n, k, j), 0.0) - A.get((n, k - 1, j), 0.0)
                    sum_term += (
                        Bij * x_glue**d2 * j * ln_x ** (j - 1)
                        - Aij * x_glue**d1 * ((d2 - d1) * ln_x**j
                        - j * ln_x ** (j - 1))
                    )
            else:
                sum_term = 0.0

            Akk_1 = A.get((n, k, k - 1), 0.0)
            Bkk_1 = B.get((n, k, k - 1), 0.0)

            term_A = Akk_1 * x_glue**d1 * (
                (k - 1) * ln_x ** (k - 2) - (d2 - d1) * ln_x ** (k - 1)
            )
            term_B = Bkk_1 * x_glue**d2 * (
                (k - 1) * ln_x ** (k - 2)) + _zeta(params, n, k, d2, x_glue)

            A[(n, k, 0)] = A.get((n, k - 1, 0), 0.0) + (x_glue ** (-d1) / (d2 - d1)) * (
                sum_term + term_A + term_B
            )

        # Compute B_{n,k,0} for k=2,...,k_max
        for k in range(2, k_max + 1):
            x_glue = x_star[n + k - 1]
            ln_x = math.log(x_glue)

            if k != 2:
                sum_term = 0.0
                for j in range(1, k - 1):
                    Aij = A.get((n, k, j), 0.0) - A.get((n, k - 1, j), 0.0)
                    Bij = B.get((n, k, j), 0.0) - B.get((n, k - 1, j), 0.0)
                    sum_term += (
                        Aij * x_glue**d1 * j * ln_x ** (j - 1)
                        - Bij * x_glue**d2 * ((d1 - d2) * ln_x**j
                        - j * ln_x ** (j - 1))
                    )
            else:
                sum_term = 0.0

            Bkk_1 = B.get((n, k, k - 1), 0.0)
            Akk_1 = A.get((n, k, k - 1), 0.0)

            term_B = Bkk_1 * x_glue**d2 * (
                (k - 1) * ln_x ** (k - 2) - (d1 - d2) * ln_x ** (k - 1)
            )
            term_A = Akk_1 * x_glue**d1 * (
                (k - 1) * ln_x ** (k - 2)) + _zeta(params, n, k, d1, x_glue)

            B[(n, k, 0)] = B.get((n, k - 1, 0), 0.0) + (x_glue ** (-d2) / (d1 - d2)) * (
                sum_term + term_B + term_A
            )

        # Now build the equation as in the corrected LaTeX with i -> n
        k_max = n_bar - n
        x_top = x_star[n_bar - 1]
        ln_xt = math.log(x_top)

        Akk0 = A.get((n, k_max - 1, 0), 0.0)

        if n != n_bar - 2:
            sum_j = 0.0
            for j in range(1, k_max - 1):
                Bij = B.get((n, k_max, j), 0.0) - B.get((n, k_max - 1, j), 0.0)
                Ajk = A.get((n, k_max - 1, j), 0.0)
                sum_j += (
                    Bij * x_top**d2 * j * ln_xt ** (j - 1)
                    + Ajk * x_top**d1 * ((d2 - d1) * ln_xt**j - j * ln_xt ** (j - 1))
                )
        else:
            sum_j = 0.0

        Bkk_last = B.get((n, k_max, k_max - 1), 0.0)
        last_term = Bkk_last * x_top**d2 * (k_max - 1) * ln_xt ** (k_max - 2)

        zeta_term = _zeta(params, n, k_max, d2, x_top)

        bracket = sum_j + last_term + zeta_term
        
        term_main = Akk0
        term_bracket = (x_top ** (-d1) / (d2 - d1)) * bracket
        F_val = term_main + term_bracket

        if verbose:
            print(f"--- Debug F_n(x) at n={n}, x={x_n} ---")
            print(f"  Term 1 (Akk0): {term_main:.6e}")
            print(f"  Term 2 (Bracket): {term_bracket:.6e}")
            print(f"  Sum: {F_val:.6e}")
            print(
                "  Cancellation check: |Sum| / max(|T1|, |T2|) = "
                f"{abs(F_val)/max(abs(term_main), abs(term_bracket)):.6e}"
            )

        return F_val

    # --- Solve thresholds starting from n = n_bar-2 down to 0 ---
    for n in range(n_bar - 2, -1, -1):
        # for level n we assume all (n+1,.,.) coeffs are ready
        compute_log_coeffs_for_level(n)

        if debug_n is not None and n == debug_n:
            # Return the function F_n(x) for inspection
            # We capture n in the lambda default argument to avoid closure issues, 
            # though n is local to the loop so it should be fine.
            return lambda x, verbose=False: F_for_level(n, x, verbose=verbose)

        x0_guess = 0.001
        root = newton(lambda x: F_for_level(n, x), x0_guess, maxiter=400)
        x_star[n] = root
        compute_A1B1_for_level(n)

    # --- Build value functions v(x,n) ---
    def make_v_for_level(n: int) -> Callable[[ModelParams, float], float]:
        k_max = n_bar - n

        def v(params_: ModelParams, x: float) -> float:
            if x < x_star[n]:
                return g_func(params_, x, n, 0)

            # Find region k such that x in [x_{n+k-1}^*, x_{n+k}^*)
            # with convention x_{n+k_max}^* = +inf.
            k_region = k_max
            for k in range(1, k_max):
                lower = x_star[n + k - 1]
                upper = x_star[n + k] if (n + k) in x_star else math.inf
                if lower <= x < upper:
                    k_region = k
                    break

            val = 0.0
            ln_x = math.log(x)
            for j in range(0, k_region):
                Aj = A.get((n, k_region, j), 0.0)
                Bj = B.get((n, k_region, j), 0.0)
                val += Aj * ln_x**j * x**d1 + Bj * ln_x**j * x**d2
            val += g_func(params_, x, n, k_region)
            return val

        return v

    solutions: Dict[int, LevelSolution] = {}
    for n in range(0, n_bar):
        solutions[n] = LevelSolution(x_star=x_star[n], v=make_v_for_level(n))

    if return_coeffs:
        return solutions, A, B

    return solutions

def V_hat_conventional(params: ModelParams, x: float) -> float:
    """Perpetuity value under the conventional model (no jumps)."""

    return x / (params.r - params.mu) - params.c / params.r


def solve_conventional_model(params: ModelParams) -> LevelSolution:
    """Solve the benchmark no-jump model (lambda=0, phi=0)."""

    params = ModelParams(**{**params.__dict__, "lambd": 0, "phi": 0})
    d_2 = d_2_func(params)

    x_star = d_2 / (d_2 - 1) * (params.c / params.r + params.L) * (params.r - params.mu)
    B = -1 / (d_2 * (params.r - params.mu)) * x_star ** (1 - d_2)

    def make_v_conventional() -> Callable[[ModelParams, float], float]:
        def v(params: ModelParams, x: float) -> float:
            if x < x_star:
                return params.L - V_hat_conventional(params, x)
            return B * x**d_2

        return v

    solution = LevelSolution(x_star=x_star, v=make_v_conventional())
    return solution