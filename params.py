from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    """Model parameters for the analytical exit problem."""

    r: float      # Discount rate
    mu: float     # Drift of GBM
    sigma: float  # Volatility of GBM
    lambd: float  # Arrival rate of Poisson process
    phi: float    # Mean jump size
    c: float      # Fixed instantaneous cost
    L: float      # Salvage value / sunk cost

    def __post_init__(self) -> None:
        # Enforce admissible parameter region for the stopping problem.
        if self.c + self.r * self.L <= 0:
            raise ValueError("Model requires c + r * L > 0.")
        if self.r == self.mu:
            raise ValueError("Model requires r != mu to avoid division by zero.")