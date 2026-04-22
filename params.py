from dataclasses import dataclass
import math
@dataclass(frozen=True)
class ModelParams:
    r: float        # Discount rate
    mu: float       # Drift of GBM
    sigma: float    # volatility of GBM
    lambd: float    # Arrival rate of Poisson process
    phi: float      # Jump size mean
    c: float        # Fixed instantaneous cost
    L: float        # Salvage value / Sunk Cost

    # @property
    # def d_1(self):
    #     return 0.5 - self.mu / self.sigma**2 + math.sqrt((self.mu / self.sigma**2 - 0.5)**2 + 2* (self.r + self.lambd) / self.sigma**2)

    # @property
    # def d_2(self):
    #     return 0.5 - self.mu / self.sigma**2 - math.sqrt((self.mu / self.sigma**2 - 0.5)**2 + 2* (self.r + self.lambd) / self.sigma**2)

    # @property
    # def S_1(self):
    #     return self.lambd/(self.r+self.lambd)

    # @property
    # def S_2(self):
    #     return self.lambd/(self.r+self.lambd-self.mu)

    def __post_init__(self) -> None:
        # Enforce admissible parameter region for the stopping problem.
        if not self.c + self.r * self.L > 0:
            raise ValueError("Model requires c > r * L.")
        if self.r == self.mu:
            raise ValueError("Cannot have r = mu as we will get division by zero")