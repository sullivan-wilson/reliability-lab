
import numpy as np
from scipy import stats

def mae(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean(np.abs(y - yhat)))

def rmse(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat)**2)))

def mspe(y, yhat, eps=1e-12):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean(((yhat - y) / (y + eps))**2))

def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return float(1 - ss_res / (ss_tot + 1e-12))

def ae(y_last, yhat_last):
    return float(abs(y_last - yhat_last))

def ks_test_residuals(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    res = y - yhat
    if len(res) < 5:
        return {"stat": float('nan'), "pvalue": float('nan')}
    res_std = (res - np.mean(res)) / (np.std(res) + 1e-12)
    stat, p = stats.kstest(res_std, 'norm')
    return {"stat": float(stat), "pvalue": float(p)}
