
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

@dataclass
class MOParams:
    lam0: float
    theta: float

class MOModel:
    def __init__(self):
        self.params = None

    def fit(self, t):
        t = np.asarray(t, dtype=float)
        T = float(t[-1])

        def nll(x):
            lam0 = np.exp(x[0])
            theta = np.exp(x[1])
            lam = lam0 / (1.0 + lam0 * theta * t)
            if np.any(lam <= 0):
                return np.inf
            mT = (1.0 / theta) * np.log(1.0 + lam0 * theta * T)
            return -np.sum(np.log(lam)) + mT

        res = minimize(nll, x0=np.log([1.0, 0.01]), method='Nelder-Mead', options=dict(maxfev=20000, maxiter=20000))
        lam0 = float(np.exp(res.x[0]))
        theta = float(np.exp(res.x[1]))
        self.params = MOParams(lam0=lam0, theta=theta)
        return self

    def m(self, t):
        t = np.asarray(t, dtype=float)
        lam0, th = self.params.lam0, self.params.theta
        return (1.0 / th) * np.log(1.0 + lam0 * th * t)

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        lam0, th = self.params.lam0, self.params.theta
        return lam0 / (1.0 + lam0 * th * t)

    def predict_cum(self, t):
        return self.m(t)
