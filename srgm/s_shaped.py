
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

@dataclass
class SParams:
    a: float
    b: float

class SShapedModel:
    def __init__(self):
        self.params = None

    def fit(self, t):
        t = np.asarray(t, dtype=float)
        n = len(t)
        T = float(t[-1])

        def nll(theta):
            a = np.exp(theta[0]) + n + 1e-6
            b = np.exp(theta[1])
            lam = a * (b**2) * t * np.exp(-b * t)
            lam[lam <= 1e-300] = 1e-300
            mT = a * (1.0 - (1.0 + b * T) * np.exp(-b * T))
            return -np.sum(np.log(lam)) + mT

        res = minimize(nll, x0=np.log([n + 1.0, 0.5]), method='Nelder-Mead', options=dict(maxfev=20000, maxiter=20000))
        a = float(np.exp(res.x[0]) + n + 1e-6)
        b = float(np.exp(res.x[1]))
        self.params = SParams(a=a, b=b)
        return self

    def m(self, t):
        t = np.asarray(t, dtype=float)
        a, b = self.params.a, self.params.b
        return a * (1.0 - (1.0 + b * t) * np.exp(-b * t))

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        a, b = self.params.a, self.params.b
        return a * (b**2) * t * np.exp(-b * t)

    def predict_cum(self, t):
        return self.m(t)
