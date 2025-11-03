
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize_scalar

@dataclass
class JMParams:
    N0: float
    phi: float

class JMModel:
    def __init__(self):
        self.params = None

    def fit(self, delta):
        delta = np.asarray(delta, dtype=float)
        n = len(delta)
        i = np.arange(1, n + 1, dtype=float)

        def phi_hat(N0):
            denom = np.sum(delta / (N0 - i + 1.0))
            return n / (denom + 1e-12)

        def nll_N0(N0):
            if N0 <= n + 1e-6:
                return np.inf
            phi = phi_hat(N0)
            rates = phi * (N0 - i + 1.0)
            if np.any(rates <= 0):
                return np.inf
            return -np.sum(np.log(rates)) + np.sum(rates * delta)

        res = minimize_scalar(nll_N0, bounds=(n + 1e-3, n + 1e3), method='bounded', options=dict(maxiter=2000))
        N0 = float(res.x)
        phi = float(phi_hat(N0))
        self.params = JMParams(N0=N0, phi=phi)
        return self

    def expected_interval(self, k):
        N0, phi = self.params.N0, self.params.phi
        return 1.0 / (phi * (N0 - k + 1.0))

    def intensity_k(self, k):
        N0, phi = self.params.N0, self.params.phi
        return phi * (N0 - k + 1.0)
