import math
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple, overload

from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from tqdm import tqdm

from plot_utils import get_figsize, save_figure, set_plot_style
from .utils import (
    ModelParams,
    n_bar_func,
    d_1_func,
    d_2_func,
    g_func,
    perpetuity_value,
    x_star_top_func,
    x_star_top_func_alt,
    zeta_func_alt,
)


@dataclass
class LevelSolution:
    """Container for one level n.

    - x_star: optimal stopping threshold x_n^*
    - v: continuation value function v(x,n)
    """

    x_star: float
    v: Callable[[ModelParams, float], float]


def _zeta(params: ModelParams, n: int, k: int, d: float, x_star_nkp1: float) -> float:
    """Wrapper implementing zeta(n,k,d) using the paper's definition.

    This mirrors utils.zeta_func_alt but uses x = x_{n+k-1}^* explicitly.
    """

    return zeta_func_alt(params, n, k, d, x_star_nkp1)


@overload
def solve_all_levels(params: ModelParams, n_bar: Optional[int] = ..., debug_n: None = ..., return_coeffs: Literal[False] = ...) -> Dict[int, LevelSolution]: ...
@overload
def solve_all_levels(params: ModelParams, n_bar: Optional[int] = ..., *, debug_n: int, return_coeffs: Literal[False] = ...) -> Callable[[float, bool], float]: ...
@overload
def solve_all_levels(params: ModelParams, n_bar: Optional[int] = ..., debug_n: None = ..., *, return_coeffs: Literal[True]) -> Tuple[Dict[int, LevelSolution], Dict[Tuple[int, int, int], float], Dict[Tuple[int, int, int], float]]: ...

def solve_all_levels(params: ModelParams, n_bar: Optional[int] = None, debug_n: Optional[int] = None, return_coeffs: bool = False):
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
    A: Dict[Tuple[int, int, int], float] = {}
    B: Dict[Tuple[int, int, int], float] = {}

    # Growth condition: for all n and j, A_{n, k_max, j} = 0
    # where k_max(n) = n_bar - n
    for n in range(0, n_bar):
        k_max = n_bar - n
        for j in range(0, k_max):
            A[(n, k_max, j)] = 0.0

    # Base coefficients for k = 1 using zeta(n,1,d)
    # A_{n,1,0} for n=0,...,n_bar-2; B_{n,1,0} for n=0,...,n_bar-1
    for n in range(0, n_bar):
        x_star_n = x_star_top_func(params) if n == n_top else None
        # We do not yet know x_n^* for n < n_top; these will be filled after
        # thresholds are solved. For now we only set B for the top level
        if n == n_top:
            zeta_n1_d1 = _zeta(params, n, 1, d1, x_star_n)
            B[(n, 1, 0)] = (x_star_n ** (-d2)) / (d1 - d2) * zeta_n1_d1
            # A_{n_top,1,0} = 0 by growth condition of base case
            A[(n, 1, 0)] = 0.0

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
            print(f"  Cancellation check: |Sum| / max(|T1|, |T2|) = {abs(F_val)/max(abs(term_main), abs(term_bracket)):.6e}")

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

        x_upper = x_star[n_bar - 1]
        x0_guess = 0.001
        root = newton(lambda x: F_for_level(n, x), x0_guess, maxiter=400)
        x_star[n] = root
        compute_A1B1_for_level(n)

    # --- Build value functions v(x,n) ---
    def make_v_for_level(n: int) -> Callable[[ModelParams, float], float]:
        k_max = n_bar - n

        def v(params_: ModelParams, x: float) -> float:
            x_arr = np.asarray(x, dtype=float)
            res = np.zeros_like(x_arr, dtype=float)

            for idx, xv in np.ndenumerate(x_arr):
                if xv < x_star[n]:
                    res[idx] = g_func(params_, xv, n, 0)
                else:
                    # find region k such that x in [x_{n+k-1}^*, x_{n+k}^*)
                    # with convention x_{n+k_max}^* = +inf
                    k_region = k_max
                    for k in range(1, k_max):
                        lower = x_star[n + k - 1]
                        upper = x_star[n + k] if (n + k) in x_star else math.inf
                        if lower <= xv < upper:
                            k_region = k
                            break

                    # compute sum over j for this region
                    val = 0.0
                    ln_x = math.log(xv)
                    for j in range(0, k_region):
                        Aj = A.get((n, k_region, j), 0.0)
                        Bj = B.get((n, k_region, j), 0.0)
                        val += (Aj * ln_x**j * xv**d1 + Bj * ln_x**j * xv**d2)
                    val += g_func(params_, xv, n, k_region)
                    res[idx] = val

            return float(res) if np.isscalar(x) else res

        return v

    solutions: Dict[int, LevelSolution] = {}
    for n in range(0, n_bar):
        solutions[n] = LevelSolution(x_star=x_star[n], v=make_v_for_level(n))

    if return_coeffs:
        return solutions, A, B

    return solutions

def V_hat_conventional(params, x):
    return x/(params.r-params.mu) - params.c/params.r

def solve_conventional_model(params):
    params = ModelParams(**{**params.__dict__, "lambd": 0, "phi": 0})
    d_1 = d_1_func(params)
    d_2 = d_2_func(params)

    x_star = d_2/(d_2-1)*(params.c/params.r + params.L)*(params.r-params.mu)
    B = -1/(d_2*(params.r-params.mu))*x_star**(1-d_2)

    def make_v_conventional():
        def v(params, x):
            x_arr = np.asarray(x, dtype=float)
            res = np.zeros_like(x_arr, dtype=float)
            for idx, xv in np.ndenumerate(x_arr):
                if xv < x_star:
                    res[idx] = params.L-V_hat_conventional(params,xv)
                else:
                    res[idx] = B*xv**d_2
            return float(res) if np.isscalar(x) else res
        return v
    solution = LevelSolution(x_star=x_star, v=make_v_conventional())
    return solution


def plot_option_vals_all_states(params: ModelParams, n_bar: int = None, x_max: float = None) -> None:
    """Plot option values v(x,n) and thresholds for all states.

    Generalises `plot_option_vals_all_states` from n_bar_3/main_solver_rita.py.
    """

    if n_bar is None:
        n_bar = n_bar_func(params)

    solutions = solve_all_levels(params, n_bar=n_bar)
    conventional_solution = solve_conventional_model(params)

    if x_max is None:
        x_max = x_star_top_func(params)+0.3

    x_vals = np.linspace(0.0, x_max, 2000)
    figsize = get_figsize()
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    param_text = "\n".join(
        [
            rf"$r = {params.r}$",
            rf"$\mu = {params.mu}$",
            rf"$\sigma = {params.sigma}$",
            rf"$\lambda = {params.lambd}$",
            rf"$\phi = {params.phi}$",
            rf"$c = {params.c}$",
            rf"$L = {params.L}$",
        ]
    )

    for n in range(0, n_bar):
        v_n = solutions[n].v
        option_vals = np.array([v_n(params, x) for x in x_vals])
        perp_vals = np.array([perpetuity_value(params, x, n) for x in x_vals])
        firm_vals = option_vals+perp_vals
        conventional_v = conventional_solution.v
        perp_conventional = np.array([V_hat_conventional(params, x) for x in x_vals])
        option_conventional = np.array([conventional_v(params, x) for x in x_vals])
        firm_conventional = perp_conventional+option_conventional
        
        if n == 0:
            alpha = 1.0
            linewidth = 1.0
            zorder = 10
            ax.plot(x_vals, perp_vals, label=r"Perpetual value $\hat{V}(x,0)$", linestyle="--", color="black", alpha=0.6, linewidth=0.7, zorder=zorder-1)
            ax.plot(x_vals, firm_vals, label=rf"Firm value V(x,{n})", alpha=alpha, linewidth=linewidth, zorder=zorder)
            # ax.plot(x_vals, firm_conventional, label=rf"Conventional Model Firm value V(x)", alpha=alpha, linewidth=linewidth, zorder=zorder)
        else:
            alpha = 0.0
            linewidth = 0.0
            zorder = 1
            ax.plot(x_vals, firm_vals, label=rf"Firm value V(x,{n})", alpha=alpha, linewidth=linewidth, zorder=zorder)


        x_star_n = solutions[n].x_star
        x_conventional = conventional_solution.x_star
        if x_star_n <= x_max:
            if n==0:
                ax.axvline(x_star_n, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
                # ax.axvline(x_conventional, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
                y_val=10
                ax.annotate(
                    "Exit threshold",
                    xy=(x_star_n, y_val / 2),
                    xytext=(x_star_n/4 + 0.5, y_val*1.5),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                    fontsize=6,
                    color="black",
                    ha="right",
                    va="center",
                )
                ax.text(
                    x_star_n+0.09,
                    ax.get_ylim()[0]+0.5,
                    rf"$x_{n}^*={x_star_n:.2f}$",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    rotation=90,
                    color="gray",
                )
            else:
                pass
                # ax.axvline(x_star_n, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("Firm value")
    # ax.set_title(r"")
    # Add descriptive annotations with arrows instead of a legend
    y_min, y_max = ax.get_ylim()
    x_min, x_max_plot = ax.get_xlim()

    # Choose an x-position reasonably far to the right but within data
    x_ref = x_max_plot * 0.7
    x_annot = x_max_plot * 0.97

    # Perpetual value annotation (more subtle styling)
    y_perp = perp_vals[np.argmin(np.abs(x_vals - x_ref))]
    ax.annotate(
        r"Firm value without option: $\hat{V}(x,0)$",
        xy=(3.12, -5.4),
        xytext=(4.9, y_max - 0.75 * (y_max - y_min)),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.6, alpha=1.0),
        fontsize=6,
        color="gray",
        ha="right",
        va="center",
    )

    # Firm value annotation (slightly below perpetuity label)
    y_firm = firm_vals[np.argmin(np.abs(x_vals - x_ref))]
    ax.annotate(
        rf"Firm value: $V(x,0)$",
        xy=(3.3, 21.5),
        xytext=(4.1, y_max - 0.15 * (y_max - y_min)),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
        fontsize=6,
        ha="right",
        va="center",
    )
    plt.tight_layout()
    # save_figure(fig, "result_1_8")
    plt.show()


def _compute_curves_general(
    params: ModelParams,
    solutions: Dict[int, LevelSolution],
    x_vals: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute option, perpetuity, and firm value curves for a given state n."""

    v_n = solutions[n].v
    perpetuity = np.array([perpetuity_value(params, x, n) for x in x_vals])
    option = np.array([v_n(params, x) for x in x_vals])
    firm = perpetuity + option
    return option, perpetuity, firm


def animate_full_transition(
    params: ModelParams,
    n_bar: int = None,
    frames_per_transition: int = 90,
    pause_frames: int = 30,
    x_max: float = None,
    simulate_gbm: bool = False,
    x0: float = 0.1,
    n_time_steps: int = 1000,
    save_gif: bool = False,
    gif_path: str = "full_transition.gif",
) -> None:
    """Animate the firm/option/perpetuity values as n increases from 0 to n_bar-1.

    This generalises the `animate_full_transition` from `n_bar_3/main_solver.py`
    to arbitrary n_bar, using the general solver in this file.
    """

    if n_bar is None:
        n_bar = n_bar_func(params)

    solutions = solve_all_levels(params, n_bar=n_bar)

    if x_max is None:
        x_max = x_star_top_func(params)*1.5

    x_vals = np.linspace(0.0, x_max, 800)

    # Precompute curves and thresholds for all states
    curves = []
    thresholds = []
    for n in range(n_bar):
        curves.append(_compute_curves_general(params, solutions, x_vals, n))
        thresholds.append(solutions[n].x_star)

    fig, ax = plt.subplots(figsize=(12, 6))
    param_text = "\n".join(
        [
            rf"$r = {params.r}$",
            rf"$\mu = {params.mu}$",
            rf"$\sigma = {params.sigma}$",
            rf"$\lambda = {params.lambd}$",
            rf"$\phi = {params.phi}$",
            rf"$c = {params.c}$",
            rf"$L = {params.L}$",
        ]
    )

    opt_0, perp_0, firm_0 = curves[0]
    lines = [
        ax.plot(x_vals, opt_0, label=r"Option value")[0],
        ax.plot(x_vals, perp_0, label=r"Perpetuity value")[0],
        ax.plot(x_vals, firm_0, label=r"Firm value")[0],
    ]
    # Moving threshold line (initially at x_0^*)
    thresh_line = ax.axvline(thresholds[0], color="gray", linestyle=":", linewidth=1.5)

    # Placeholder for GBM marker and path, initialised after frame count
    gbm_marker = None
    gbm_path = None

    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("x")
    title = ax.set_title(r"State $n=0$")

    # Total frames: transitions between every consecutive pair plus final pause
    num_transitions = max(n_bar - 1, 0)
    total_frames = 0
    if num_transitions > 0:
        total_frames = num_transitions * frames_per_transition + (num_transitions - 1) * pause_frames
    if total_frames <= 0:
        total_frames = 1

    # Optional GBM simulation: moving marker along the x-axis (state variable)
    if simulate_gbm:
        T_max = 1.0 / params.lambd * max(n_bar, 1)
        t_grid = np.linspace(0.0, T_max, n_time_steps + 1)

        # Exponential inter-arrival times with mean 1/lambda
        jump_times = [0.0]
        current_time = 0.0
        rng = np.random.default_rng()
        while current_time < T_max:
            current_time += rng.exponential(1.0 / params.lambd)
            if current_time <= T_max:
                jump_times.append(current_time)

        X = np.empty_like(t_grid)
        X[0] = x0
        mu = params.mu
        sigma = params.sigma

        for k in range(1, len(t_grid)):
            dt_k = t_grid[k] - t_grid[k - 1]
            dW = rng.normal(0.0, math.sqrt(dt_k))
            X[k] = X[k - 1] * math.exp((mu - 0.5 * sigma**2) * dt_k + sigma * dW)

        # Map animation frames to GBM time grid
        if len(t_grid) > 1:
            idx_per_frame = max(len(t_grid) // total_frames, 1)
        else:
            idx_per_frame = 1

        # Store for use in update via closure
        gbm_path = X
        gbm_marker, = ax.plot([gbm_path[0]], [0.0], marker="o", color="black", markersize=6, label="GBM state x_t")

    def update(frame: int):
        artists = list(lines) + [thresh_line]
        if simulate_gbm and gbm_marker is not None:
            artists.append(gbm_marker)

        if num_transitions == 0:
            return artists

        # Determine which transition we are in and local frame index
        transition_block = frames_per_transition + pause_frames
        t = frame // transition_block
        local = frame % transition_block

        # Cap at last transition
        if t >= num_transitions:
            t = num_transitions - 1
            local = frames_per_transition + pause_frames - 1

        n_from = t
        n_to = t + 1
        opt_from, perp_from, firm_from = curves[n_from]
        opt_to, perp_to, firm_to = curves[n_to]

        if local < frames_per_transition:
            # Blend between n_from and n_to
            w = local / (frames_per_transition - 1) if frames_per_transition > 1 else 1.0
            blended = [
                (1 - w) * opt_from + w * opt_to,
                (1 - w) * perp_from + w * perp_to,
                (1 - w) * firm_from + w * firm_to,
            ]
            title.set_text(rf"Transition from $n={n_from} \to n={n_to}$")
            # Move threshold line smoothly between x_{n_from}^* and x_{n_to}^*
            x_thresh = (1 - w) * thresholds[n_from] + w * thresholds[n_to]
            thresh_line.set_xdata([x_thresh, x_thresh])
            for line, y in zip(lines, blended):
                line.set_ydata(y)
        else:
            # Pause at n_to
            title.set_text(rf"State $n={n_to}$")
            for line, y in zip(lines, (opt_to, perp_to, firm_to)):
                line.set_ydata(y)
            # Keep threshold line fixed at x_{n_to}^*
            x_thresh = thresholds[n_to]
            thresh_line.set_xdata([x_thresh, x_thresh])

        # Update GBM marker position along x-axis
        if simulate_gbm and gbm_marker is not None:
            frame_idx = min(frame * idx_per_frame, len(gbm_path) - 1)
            x_gbm = gbm_path[frame_idx]
            gbm_marker.set_data([x_gbm], [0.0])

        return artists

    ani = FuncAnimation(fig, update, frames=total_frames, interval=40, blit=False)
    handles, labels = ax.get_legend_handles_labels()
    param_handle = Line2D([], [], linestyle="none", marker="", color="none")
    handles.append(param_handle)
    labels.append(param_text)
    ax.legend(handles, labels, loc="upper left", handletextpad=0.1)
    plt.tight_layout()

    if save_gif:
        try:
            ani.save(gif_path, writer="pillow", fps=int(1000/40))
        except Exception as exc:  # pragma: no cover
            print(f"Failed to save GIF to {gif_path}: {exc}")

    plt.show()


def plot_x0_star_heatmap(
    base_params: ModelParams,
    lambda_values: np.ndarray,
    phi_values: np.ndarray,
    n_bar: int | None = None,
) -> None:
    """Plot heatmap of x_0^* as a function of (lambda, phi).

    Parameters
    ----------
    base_params : ModelParams
        Baseline parameters; ``lambd`` and ``phi`` are varied over the grids.
    lambda_values : np.ndarray
        1D array of arrival rates to put on the x-axis.
    phi_values : np.ndarray
        1D array of impact severities to put on the y-axis.
    n_bar : int, optional
        If given, use this n_bar instead of recomputing it each time.
    """

    lambda_values = np.asarray(lambda_values, dtype=float)
    phi_values = np.asarray(phi_values, dtype=float)

    Z = np.empty((len(phi_values), len(lambda_values)))

    total = len(phi_values) * len(lambda_values)
    for idx in tqdm(range(total), desc="Computing x_0^* heatmap"):
        i, j = divmod(idx, len(lambda_values))
        phi = phi_values[i]
        lambd = lambda_values[j]
        params = ModelParams(
            r=base_params.r,
            mu=base_params.mu,
            sigma=base_params.sigma,
            lambd=lambd,
            phi=phi,
            c=base_params.c,
            L=base_params.L,
        )
        sols = solve_all_levels(params, n_bar=n_bar)
        Z[i, j] = sols[0].x_star
    figsize = get_figsize()
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[lambda_values.min(), lambda_values.max(), phi_values.min(), phi_values.max()],
        cmap="viridis",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Exit threshold $x_0^*$")

    ax.set_xlabel(r"Arrival rate $\lambda$")
    ax.set_ylabel(r"Impact severity $\phi$")
    ax.set_title(r"Heatmap of $x_0^*$ over $(\lambda, \phi)$")

    # Set ticks to include start and end values
    num_ticks = 6
    x_ticks = np.linspace(lambda_values.min(), lambda_values.max(), num_ticks)
    y_ticks = np.linspace(phi_values.min(), phi_values.max(), num_ticks)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Format ticks to 2 decimal places
    ax.set_xticklabels([f"{x:.2f}" for x in x_ticks])
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])

    plt.tight_layout()
    save_figure(fig, filename="result_3_1")
    plt.show()

def plot_x_thresholds_and_revenue_growth_vs_number_of_shocks(params: ModelParams, params2: ModelParams = None):
    """
    Computes and plots x_n^* and revenue growth x0 * exp(mu * t)
    vs shock number n. Includes parameter legend.
    """

    # --- Solve model 1 ---
    n_bar = n_bar_func(params)
    solutions = solve_all_levels(params, n_bar=n_bar)
    x_stars = [solutions[n].x_star for n in range(n_bar)]
    
    # Add infinite threshold for the next state
    x_stars.append(np.inf)
    n_vals = np.arange(len(x_stars))
    x0 = x_stars[0]
    
    if params.mu < params.lambd:
        rev_vals = x0 * (1 - params.mu / params.lambd) ** (-n_vals)
    else:
        time_vals = n_vals / params.lambd
        rev_vals = x0 * np.exp(params.mu * time_vals)

    # --- Solve model 2 (if provided) ---
    x_stars2 = None
    rev_vals2 = None
    n_vals2 = None
    if params2:
        n_bar2 = n_bar_func(params2)
        solutions2 = solve_all_levels(params2, n_bar=n_bar2)
        x_stars2 = [solutions2[n].x_star for n in range(n_bar2)]
        
        # Add infinite threshold for the next state
        x_stars2.append(np.inf)
        n_vals2 = np.arange(len(x_stars2))
        x0_2 = x_stars2[0]
        
        if params2.mu < params2.lambd:
            rev_vals2 = x0_2 * (1 - params2.mu / params2.lambd) ** (-n_vals2)
        else:
            time_vals2 = n_vals2 / params2.lambd
            rev_vals2 = x0_2 * np.exp(params2.mu * time_vals2)

    # --- Plot ---
    set_plot_style()
    
    # ----------------------- PARAMETER LEGEND -----------------------
    def get_param_text(p):
        return "\n".join(
            [
                rf"$r = {p.r}$",
                rf"$\mu = {p.mu}$",
                rf"$\sigma = {p.sigma}$",
                rf"$\lambda = {p.lambd}$",
                rf"$\phi = {p.phi}$",
                rf"$c = {p.c}$",
                rf"$L = {p.L}$",
            ]
        )

    if params2:
        # Create 2 subplots side-by-side fitting in the text width
        figsize = get_figsize(width_fraction=1.0, aspect_ratio=2.2)
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        ax1, ax2 = axes
    else:
        figsize = get_figsize(aspect_ratio=1.2)
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    from matplotlib.ticker import MaxNLocator

    # Calculate plot limits and adjust infinite thresholds
    # Filter out infinity for max calculation
    finite_x = [x for x in x_stars if x < np.inf]
    max_val = max(np.max(rev_vals), max(finite_x) if finite_x else 0)
    
    if params2:
        finite_x2 = [x for x in x_stars2 if x < np.inf]
        max_val2 = max(np.max(rev_vals2), max(finite_x2) if finite_x2 else 0)
        max_val = max(max_val, max_val2)
    
    y_limit = max_val * 1.1
    
    # Replace infinity with a value larger than y_limit for plotting
    large_val = y_limit * 1.5
    x_stars_plot = [x if x < np.inf else large_val for x in x_stars]
    if params2:
        x_stars2_plot = [x if x < np.inf else large_val for x in x_stars2]

    # Plot Set 1
    color1 = "tab:blue"
    label1 = rf"$\lambda = {params.lambd}$"
    ax1.set_title(label1)
    ax1.plot(n_vals, x_stars_plot, drawstyle="steps-post", marker="o", linewidth=1, label=r"Exit threshold $x_n^*$", color=color1)
    ax1.plot(n_vals, rev_vals, linestyle="--", linewidth=1, label=r"Expected revenue", color=color1)
    
    ax1.set_ylim(bottom=0, top=y_limit)
    
    ax1.set_xlabel(r"$n$")
    ax1.set_ylabel(r"Revenue")
    ax1.yaxis.set_tick_params(labelleft=True)
    ax1.legend(loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add param text to ax1
    param_text1 = get_param_text(params)
    # ax1.text(0.95, 0.95, param_text1, transform=ax1.transAxes, fontsize=8,
    #         verticalalignment='top', horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot Set 2
    if params2 and ax2:
        color2 = "tab:orange"
        label2 = rf"$\lambda = {params2.lambd}$"
        ax2.set_title(label2)
        ax2.plot(n_vals2, x_stars2_plot, drawstyle="steps-post", marker="s", linewidth=1, label=r"Exit threshold $x_n^*$", color=color2)
        ax2.plot(n_vals2, rev_vals2, linestyle="--", linewidth=1, label=r"Expected revenue", color=color2)
        
        ax2.set_ylim(bottom=0, top=y_limit)
        
        ax2.set_xlabel(r"$n$")
        # ax2.set_ylabel(r"Revenue") # Optional
        ax2.legend(loc="upper left")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add param text to ax2
        param_text2 = get_param_text(params2)
        # ax2.text(0.95, 0.95, param_text2, transform=ax2.transAxes, fontsize=8,
        #         verticalalignment='top', horizontalalignment='right',
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide y-labels on ax2 to mimic sharey=True
        ax2.tick_params(labelleft=False)

    plt.tight_layout()
    # save_figure(fig, "result_4_6")
    plt.show()

def plot_exit_threshold_r_latex(
    r_values,
    fixed_params,
    phi: float = 0.2,
    L_values=None,
    n_selected: int = 0,
    save_path: str | None = None,
):
    """LaTeX-optimized plot of $x_{n}^*$ vs $r$ for fixed $\\phi$.

    - Uses paper.mplstyle via `set_plot_style`.
    - Plots three lines for the provided L values on one axis.

    Parameters
    ----------
    r_values : array-like
        Grid of risk-free rates $r$ on the x-axis.
    fixed_params : dict
        Baseline parameter dictionary; `r`, `phi`, and `L` are overwritten.
    phi : float, default 0.2
        Shock size parameter to fix across all curves.
    L_values : list[float] | None
        L-values to compare. Defaults to [-10, 0, 10].
    n_selected : int, default 0
        Shock level index $n$ for the exit threshold $x_n^*$.
    save_path : str | None
        Optional filename (without folder) to save via `save_figure`.
    """

    if L_values is None:
        L_values = [-10.0, 0.0, 10.0]

    set_plot_style("paper.mplstyle")

    # Figure size: full text width, moderate height
    figsize = get_figsize(width_fraction=1.0, height_fraction=0.6)
    fig, ax = plt.subplots(figsize=figsize)

    # cmap = cm.get_cmap("tab10")
    cmap = plt.get_cmap("tab10")
    
    # Get phi from fixed_params if available, else default to 0
    phi = fixed_params.get("phi", 0.0)

    # Store curves for annotation
    curves = {}

    for idx, L in enumerate(L_values):
        thresholds = []
        for r in r_values:
            try:
                param_dict = fixed_params.copy()
                param_dict["phi"] = phi
                param_dict["L"] = L
                param_dict["r"] = r

                params = ModelParams(**param_dict)

                n_bar = n_bar_func(params)
                if n_selected >= n_bar:
                    thresholds.append(np.nan)
                    continue

                # solutions = solve_all_levels(params, n_bar=n_bar)
                solution = solve_conventional_model(params)
                thresholds.append(solution.x_star)
                # thresholds.append(solutions[n_selected].x_star)
            except Exception as e:
                # print(e)
                thresholds.append(np.nan)

        thresholds = np.array(thresholds, dtype=float)
        
        # Store for annotation
        curves[L] = (r_values, thresholds)
        
        ax.plot(
            r_values,
            thresholds,
            label=rf"$L = {L}$",
            color=cmap(idx),
        )

    # Axis labels and title (LaTeX-ready)
    ax.set_xlabel(r"Risk-free rate $r$")
    ax.set_ylabel(rf"Exit threshold $x_F^*$")
    ax.set_ylim(0,2.8)
    # ax.set_title(rf"Exit threshold $x_{{{n_selected}}}^*$ vs. $r$ (fixed $\phi={phi}$)")

    # Minimal styling for paper
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add text annotations with arrows for each L-curve instead of a legend
    # Use the last finite point of each curve as the arrow target.
    
    # Estimate y_span from data
    all_thresholds = np.concatenate([c[1] for c in curves.values()])
    valid_thresholds = all_thresholds[np.isfinite(all_thresholds)]
    if len(valid_thresholds) > 0:
        y_min, y_max = np.min(valid_thresholds), np.max(valid_thresholds)
        y_span = y_max - y_min
    else:
        y_span = 1.0

    for idx, L in enumerate(L_values):
        if L not in curves:
            continue
            
        r_curve_all, x_curve_all = curves[L]
        finite_mask = np.isfinite(x_curve_all)
        if not np.any(finite_mask):
            continue

        r_curve = np.array(r_curve_all)[finite_mask]
        x_curve = x_curve_all[finite_mask]

        # Take a point towards the right of the curve for annotation.
        idx_target = int(0.7 * (len(r_curve) - 1))
        r_target = r_curve[idx_target]
        x_target = x_curve[idx_target]

        # Horizontal offset for the text (in data coordinates)
        r_text = r_target + 0.02 * (r_values[-1] - r_values[0])

        # Vertical placement
        if L > 0:
            x_text = x_target - 0.15 * y_span
            label = rf"$L={L}$ (Salvage value)"
        elif L < 0:
            x_text = x_target + 0.15 * y_span
            label = rf"$L={L}$ (Exit cost)"
        else:
            x_text = x_target - 0.06 * y_span
            label = rf"$L={L}$"

        ax.annotate(
            label,
            xy=(r_target, x_target),
            xytext=(r_text, x_text),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
            ha="left",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)

    plt.show()


if __name__ == "__main__":

    params = ModelParams(
        r=0.05,mu=.02,sigma=0.1,lambd=0.1,phi=0.2,c=2,L=-10
    )
    params2 = ModelParams(
        r=0.05,mu=.02,sigma=0.1,lambd=0.05,phi=0.2,c=2,L=-10
    )
    
    # Result_1_1
    plot_option_vals_all_states(params, x_max=6)

    # plot_exit_threshold_r_latex(r_values=np.linspace(0.0201,0.14,100), fixed_params=params.__dict__, n_selected=0, 
    #                             save_path="threshold_vs_r_benchmark_3"
    #                             )
    
    # plot_option_vals_all_states(params)
    
    # solutions = solve_all_levels(params)
    # n_bar = n_bar_func(params)
    # x_stars = []
    # x_top = x_star_top_func(params)
    # option_vals_top = []
    # for n in range(n_bar):
    #     x_star_n = solutions[n].x_star
    #     v_n = solutions[n].v
    #     option_vals_top.append(v_n(params, x_top))
    #     x_stars.append(x_star_n)
    # x_vals = np.arange(n_bar)
    # plt.plot(x_vals, option_vals_top)
    # plt.plot(x_vals, x_stars)
    # plt.show()


    # lambdas = np.linspace(0.04, 1.0, 100)
    # phis = np.linspace(0.04, 1.0, 100)
    # plot_x0_star_heatmap(params, lambdas, phis)


    # plot_x_thresholds_and_revenue_growth_vs_number_of_shocks(params, params2)

    # plot_option_vals_all_states(params)
    # animate_full_transition(
    #     params,
    #     simulate_gbm=True,
    #     x0=2.5,
    #     n_time_steps=1000,
    #     save_gif=False,
    #     gif_path="full_transition_with_gbm_path.gif",
    # )