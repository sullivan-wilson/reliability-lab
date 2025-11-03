
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

@dataclass
class GOParams:
    a: float
    b: float

class GOModel:
    def __init__(self):
        self.params = None

    def fit(self, t):
        t = np.asarray(t, dtype=float)
        n = len(t)
        T = float(t[-1])

        def nll(theta):
            a = np.exp(theta[0]) + n + 1e-6
            b = np.exp(theta[1])
            lam = a * b * np.exp(-b * t)
            if np.any(lam <= 0):
                return np.inf
            mT = a * (1 - np.exp(-b * T))
            return -np.sum(np.log(lam)) + mT

        x0 = np.log([n + 1.0, 1.0])
        res = minimize(nll, x0=x0, method='Nelder-Mead', options=dict(maxfev=20000, maxiter=20000))
        a = float(np.exp(res.x[0]) + n + 1e-6)
        b = float(np.exp(res.x[1]))
        self.params = GOParams(a=a, b=b)
        return self

    def m(self, t):
        t = np.asarray(t, dtype=float)
        a, b = self.params.a, self.params.b
        return a * (1 - np.exp(-b * t))

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        a, b = self.params.a, self.params.b
        return a * b * np.exp(-b * t)

    def predict_cum(self, t):
        return self.m(t)
