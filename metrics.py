
import numpy as np
from scipy import stats
from scipy.stats import kstest

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
    """
    绝对误差：
    - 如果传入的是标量，返回单点绝对误差；
    - 如果传入的是数组，返回平均绝对误差（和 MAE 一致，但名字保留 AE）。
    """
    y_last = np.asarray(y_last)
    yhat_last = np.asarray(yhat_last)
    return float(np.mean(np.abs(y_last - yhat_last)))

def ks_test_residuals(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    res = y - yhat
    if len(res) < 5:
        return {"stat": float('nan'), "pvalue": float('nan')}
    res_std = (res - np.mean(res)) / (np.std(res) + 1e-12)
    stat, p = stats.kstest(res_std, 'norm')
    return {"stat": float(stat), "pvalue": float(p)}

def ks_on_u_values(u_values):
    """
    对 U 序列做 K-S 拟合优度检验：
    原假设 H0: U ~ Uniform(0, 1)
    返回 dict: {"stat": D统计量, "pvalue": p值}
    """
    u = np.asarray(u_values, dtype=float)
    # 避免取到 0 或 1，影响数值稳定性
    u = np.clip(u, 1e-8, 1 - 1e-8)

    stat, pvalue = kstest(u, "uniform")
    return {"stat": float(stat), "pvalue": float(pvalue)}
