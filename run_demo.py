
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_times, cum_to_intervals, train_valid_split, ensure_output_dir
from metrics import mae, rmse, mspe, r2, ae, ks_test_residuals

from srgm.go import GOModel
from srgm.jm import JMModel
from srgm.mo import MOModel
from srgm.s_shaped import SShapedModel

from timeseries.gm11 import GM11
def gm11_walk_predict_intervals(d, min_window=8):
    """
    用 GM(1,1) 对“错误间隔序列”做逐步滚动的一步预测，返回每一步的预测间隔。
    前 min_window 个点做热启动（直接用真实值），之后每一步用最近窗口重新拟合再预测下一步。
    """
    from timeseries.gm11 import GM11
    d = np.asarray(d, dtype=float)
    n = len(d)
    pred = np.empty(n, dtype=float)
    # 热启动：前几个点用真实值，避免样本太少（GM11 至少需要 4 个点）
    k0 = max(4, int(min_window))
    pred[:k0] = d[:k0]
    for k in range(k0, n):
        win = d[max(0, k - min_window):k]
        if len(win) < 4:
            pred[k] = d[k]
        else:
            gm = GM11().fit(win)
            pred[k] = max(1e-6, float(gm.predict_next()))
    return pred

def counts_from_pred_times(t_hat, t_query):
    """
    给定模型预测的'故障发生时刻序列' t_hat（长度 N），
    以及一组查询时刻 t_query（长度 M，来自真实数据的 t_k），
    返回“到每个 t_query 时刻为止，模型预计发生了多少起故障”的累计数（可与 k=1..M 对齐做指标）。
    """
    t_hat = np.asarray(t_hat, dtype=float)
    t_query = np.asarray(t_query, dtype=float)
    # searchsorted(..., right) = 有多少 t_hat <= t_query
    return np.searchsorted(t_hat, t_query, side='right').astype(float)

def fit_and_eval_on_cum(model, t_train, t_valid):
    t_all = np.concatenate([t_train, t_valid])
    model.fit(t_train)

    y_true_train = np.arange(1, len(t_train) + 1, dtype=float)
    y_hat_train = model.predict_cum(t_train)

    y_true_valid = np.arange(len(t_train) + 1, len(t_all) + 1, dtype=float)
    y_hat_valid = model.predict_cum(t_all)[len(t_train):]

    metrics_train = {
        "MAE": mae(y_true_train, y_hat_train),
        "RMSE": rmse(y_true_train, y_hat_train),
        "MSPE": mspe(y_true_train, y_hat_train),
        "R2": r2(y_true_train, y_hat_train)
    }
    metrics_valid = {
        "MAE": mae(y_true_valid, y_hat_valid),
        "RMSE": rmse(y_true_valid, y_hat_valid),
        "MSPE": mspe(y_true_valid, y_hat_valid),
        "R2": r2(y_true_valid, y_hat_valid),
        "AE_last": ae(y_true_valid[-1], y_hat_valid[-1])
    }
    ks_train = ks_test_residuals(y_true_train, y_hat_train)
    ks_valid = ks_test_residuals(y_true_valid, y_hat_valid)

    return {
        "metrics_train": metrics_train,
        "metrics_valid": metrics_valid,
        "ks_train": ks_train,
        "ks_valid": ks_valid,
        "y_true_train": y_true_train,
        "y_hat_train": y_hat_train,
        "y_true_valid": y_true_valid,
        "y_hat_valid": y_hat_valid
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/ntds_sample.csv")
    parser.add_argument("--train_n", type=int, default=28)
    parser.add_argument("--models", type=str, default="GO,JM,MO,S,GM11")
    args = parser.parse_args()

    out_dir = ensure_output_dir("output")

    t = load_times(args.csv)
    t_train, t_valid = train_valid_split(t, train_n=args.train_n)
    d_train = cum_to_intervals(t_train)

    selected = [m.strip().upper() for m in args.models.split(",")]

    results = {}

    if "GO" in selected:
        go = GOModel()
        results["GO"] = fit_and_eval_on_cum(go, t_train, t_valid)

    if "JM" in selected:
        jm = JMModel()
        jm.fit(d_train)
        lam_k = np.array([jm.intensity_k(k) for k in range(1, len(t_train)+1)], dtype=float)
        y_hat_train = np.cumsum(lam_k / (np.max(lam_k) + 1e-12))
        y_true_train = np.arange(1, len(t_train)+1, dtype=float)
        extra_k = len(t_valid)
        lam_k_val = np.array([jm.intensity_k(k) for k in range(len(t_train)+1, len(t_train)+extra_k+1)], dtype=float)
        y_hat_valid = y_hat_train[-1] + np.cumsum(lam_k_val / (np.max(lam_k) + 1e-12))
        y_true_valid = np.arange(len(t_train)+1, len(t_train)+extra_k+1, dtype=float)

        results["JM"] = {
            "metrics_train": {
                "MAE": float(np.mean(np.abs(y_true_train - y_hat_train))),
                "RMSE": float(np.sqrt(np.mean((y_true_train - y_hat_train)**2))),
                "MSPE": float(np.mean(((y_hat_train - y_true_train)/(y_true_train+1e-12))**2)),
                "R2": float(1 - np.sum((y_true_train - y_hat_train)**2)/(np.sum((y_true_train - np.mean(y_true_train))**2)+1e-12))
            },
            "metrics_valid": {
                "MAE": float(np.mean(np.abs(y_true_valid - y_hat_valid))),
                "RMSE": float(np.sqrt(np.mean((y_true_valid - y_hat_valid)**2))),
                "MSPE": float(np.mean(((y_hat_valid - y_true_valid)/(y_true_valid+1e-12))**2)),
                "R2": float(1 - np.sum((y_true_valid - y_hat_valid)**2)/(np.sum((y_true_valid - np.mean(y_true_valid))**2)+1e-12)),
                "AE_last": float(abs(y_true_valid[-1] - y_hat_valid[-1]))
            },
            "ks_train": {"stat": float('nan'), "pvalue": float('nan')},
            "ks_valid": {"stat": float('nan'), "pvalue": float('nan')},
            "y_true_train": y_true_train, "y_hat_train": y_hat_train,
            "y_true_valid": y_true_valid, "y_hat_valid": y_hat_valid,
        }

    if "MO" in selected:
        mo = MOModel()
        results["MO"] = fit_and_eval_on_cum(mo, t_train, t_valid)

    if "S" in selected or "S_SHAPED" in selected:
        s = SShapedModel()
        results["S"] = fit_and_eval_on_cum(s, t_train, t_valid)

    # if "GM11" in selected:
    #     gm = GM11()
    #     d_all = cum_to_intervals(t)
    #     gm.fit(d_all[:args.train_n])
    #     xhat_val = []
    #     cur_series = d_all[:args.train_n].tolist()
    #     for _ in range(len(t) - args.train_n):
    #         gm.fit(np.array(cur_series[-min(8, len(cur_series)):]))  # use recent 8 points
    #         nxt = float(gm.predict_next())
    #         xhat_val.append(max(nxt, 1e-6))
    #         cur_series.append(xhat_val[-1])
    #
    #     y_true_valid = np.arange(len(t_train)+1, len(t)+1, dtype=float)
    #     y_hat_valid = np.linspace(len(t_train)+1, len(t), num=len(y_true_valid))
    #     y_true_train = np.arange(1, len(t_train)+1, dtype=float)
    #     y_hat_train = np.linspace(1, len(t_train), num=len(y_true_train))
    #
    #     results["GM11"] = {
    #         "metrics_train": {
    #             "MAE": float(np.mean(np.abs(y_true_train - y_hat_train))),
    #             "RMSE": float(np.sqrt(np.mean((y_true_train - y_hat_train)**2))),
    #             "MSPE": float(np.mean(((y_hat_train - y_true_train)/(y_true_train+1e-12))**2)),
    #             "R2": float(1 - np.sum((y_true_train - y_hat_train)**2)/(np.sum((y_true_train - np.mean(y_true_train))**2)+1e-12))
    #         },
    #         "metrics_valid": {
    #             "MAE": float(np.mean(np.abs(y_true_valid - y_hat_valid))),
    #             "RMSE": float(np.sqrt(np.mean((y_true_valid - y_hat_valid)**2))),
    #             "MSPE": float(np.mean(((y_hat_valid - y_true_valid)/(y_true_valid+1e-12))**2)),
    #             "R2": float(1 - np.sum((y_true_valid - y_hat_valid)**2)/(np.sum((y_true_valid - np.mean(y_true_valid))**2)+1e-12)),
    #             "AE_last": float(abs(y_true_valid[-1] - y_hat_valid[-1]))
    #         },
    #         "ks_train": {"stat": float('nan'), "pvalue": float('nan')},
    #         "ks_valid": {"stat": float('nan'), "pvalue": float('nan')},
    #         "y_true_train": y_true_train, "y_hat_train": y_hat_train,
    #         "y_true_valid": y_true_valid, "y_hat_valid": y_hat_valid,
    #     }

    for name, res in results.items():
        print("="*60)
        print(name)
        print("TRAIN:", res["metrics_train"])
        print("VALID:", res["metrics_valid"])
        print("KS(train):", res["ks_train"], " KS(valid):", res["ks_valid"])

    fig = plt.figure()
    t_all = np.concatenate([t_train, t_valid])
    x_idx = np.arange(1, len(t_all)+1, dtype=float)

    plt.plot(x_idx, x_idx, label="Truth")
    for name, res in results.items():
        yh = np.concatenate([res["y_hat_train"], res["y_hat_valid"]])
        plt.plot(x_idx, yh, label=name)
    plt.xlabel("Failure index")
    plt.ylabel("Cumulative failures (model vs truth)")
    plt.legend()
    fig.savefig(f"{out_dir}/fit_plot.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_dir}/fit_plot.png")

if __name__ == "__main__":
    main()
