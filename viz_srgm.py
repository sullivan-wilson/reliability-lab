# viz_srgm.py
# 专门放 SRGM 课程设计用的可视化与诊断函数

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import streamlit as st


# ====== 全局中文字体配置（云端 + 本地统一） ======
def setup_chinese_font():
    # 字体文件相对路径
    base_dir = os.path.dirname(__file__)
    font_path = os.path.join(base_dir, "fonts", "NotoSansSC-Regular.ttf")  # 改成你实际的文件名

    # 把字体注册到 matplotlib
    font_manager.fontManager.addfont(font_path)
    rcParams["font.family"] = "Noto Sans SC"      # 和字体内部名称保持一致
    rcParams["axes.unicode_minus"] = False        # 解决负号显示成方块的问题

# 导入模块时就执行一次
setup_chinese_font()


# ========== 指标表 ==========

def build_metrics_table(results):
    """
    把每个模型的训练 / 验证指标拼成一个 DataFrame。
    约定 results[name]["metrics_train"] / ["metrics_valid"] 为一个 dict，
    里面包含 MAE / AE / RMSE / MSPE / R2 / ESS 等。
    """
    rows = []
    for name, res in results.items():
        mt = res["metrics_train"]
        mv = res["metrics_valid"]
        rows.append({
            "模型": name,
            "Train_MAE": mt.get("MAE"),
            "Train_AE": mt.get("AE"),
            "Train_RMSE": mt.get("RMSE"),
            "Train_MSPE": mt.get("MSPE"),
            "Train_R2": mt.get("R2"),
            "Train_ESS": mt.get("ESS"),
            "Valid_MAE": mv.get("MAE"),
            "Valid_AE": mv.get("AE"),
            "Valid_RMSE": mv.get("RMSE"),
            "Valid_MSPE": mv.get("MSPE"),
            "Valid_R2": mv.get("R2"),
            "Valid_ESS": mv.get("ESS"),
        })
    df = pd.DataFrame(rows)
    return df.set_index("模型")


# ========== 拟合曲线 m(t) ==========

def plot_cum_failures(results, t_train, t_valid):
    """
    累计失效曲线 m(t)：对应课件里“经典 SRGM 拟合曲线”的那张图。
    蓝色阶梯：真实累计故障数；彩色曲线：各模型 m(t) 预测。
    """
    model_order = ["GO", "MO", "S"]
    active = [m for m in model_order if m in results]
    if not active:
        st.info("当前未选择 GO / MO / S 模型，无法绘制累计失效曲线。")
        return None

    t_all = np.concatenate([t_train, t_valid])
    n_all = len(t_all)
    k_idx = np.arange(1, n_all + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))

    # 真实值：阶梯线
    ax.step(t_all, k_idx, where="post", label="Truth", linewidth=2)

    # 各模型预测曲线
    for name in active:
        yh_train = np.asarray(results[name]["y_hat_train"], dtype=float)
        yh_valid = np.asarray(results[name]["y_hat_valid"], dtype=float)
        yh_all = np.concatenate([yh_train, yh_valid])
        ax.plot(t_all, yh_all, label=name)

    # 训练 / 验证分割线
    t_split = t_train[-1]
    ax.axvline(t_split, linestyle="--", linewidth=1, alpha=0.6)
    ax.text(t_split, ax.get_ylim()[1],
            " Train / Valid", rotation=90,
            va="top", ha="left", fontsize=8, alpha=0.7)

    ax.set_xlabel("时间 t")
    ax.set_ylabel("累计故障数 m(t)")
    ax.set_title("累计失效曲线：模型 vs 真实")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


# ========== U 图 / Y 图相关 ==========
def compute_u_sequence(res):
    """
    从某个模型的预测构造 U 序列：
    用 yh_all / yh_all[-1] 近似作为 CDF 值 F(t_i)。
    """
    yh_train = np.asarray(res["y_hat_train"], dtype=float)
    yh_valid = np.asarray(res["y_hat_valid"], dtype=float)
    yh_all = np.concatenate([yh_train, yh_valid])
    m_end = yh_all[-1]
    if m_end <= 0:
        return None
    u = yh_all / m_end
    return np.clip(u, 1e-8, 1 - 1e-8)


def plot_u_y(u_values, title_prefix=""):
    """
    根据 U 值画 U 图和 Y 图：
    - U 图：empirical quantile vs theoretical U(0,1)
    - Y 图：-ln(1-U) 变换后与 45° 线对比
    """
    u = np.sort(np.asarray(u_values, dtype=float))
    n = len(u)
    emp = np.arange(1, n + 1) / (n + 1.0)

    # --- U 图 ---
    fig_u, ax_u = plt.subplots(figsize=(5, 4))
    ax_u.plot(emp, u, "o", markersize=4, label="样本 U(i)")
    ax_u.plot([0, 1], [0, 1], "--", label="理论 U(0,1)")
    ax_u.set_xlabel("理论分位 i/(n+1)")
    ax_u.set_ylabel("样本分位 U(i)")
    ax_u.set_title(f"{title_prefix} - U 图")
    ax_u.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax_u.legend()
    fig_u.tight_layout()

    # --- Y 图 ---
    eps = 1e-8
    x = -np.log(1 - emp + eps)
    y = -np.log(1 - u + eps)
    max_xy = max(x.max(), y.max())

    fig_y, ax_y = plt.subplots(figsize=(5, 4))
    ax_y.plot(x, y, "o", markersize=4, label="样本点")
    ax_y.plot([0, max_xy], [0, max_xy], "--", label="45° 参考线")
    ax_y.set_xlabel("理论 -ln(1-u)")
    ax_y.set_ylabel("样本 -ln(1-U)")
    ax_y.set_title(f"{title_prefix} - Y 图")
    ax_y.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax_y.legend()
    fig_y.tight_layout()

    return fig_u, fig_y


# ========== 最小相关误差：残差相关系数 ==========
def compute_resid_corr(y_true_all, y_pred_all):
    """
    计算残差的一阶自相关系数 ρ，用于“最小相关误差”检验。
    """
    y_true_all = np.asarray(y_true_all, dtype=float)
    y_pred_all = np.asarray(y_pred_all, dtype=float)
    e = y_true_all - y_pred_all
    if len(e) < 2:
        return np.nan
    return float(np.corrcoef(e[1:], e[:-1])[0, 1])


# ========== PLR：累计 log-likelihood 序列 ==========
def compute_plr_loglik(t_all, y_pred_all):
    """
    简化版 PLR：基于预测的累计均值 m(t)，构造每个区间的 λ_k，
    计算 log-likelihood 增量并累加得到 loglik 序列。
    """
    t_all = np.asarray(t_all, dtype=float)
    m = np.asarray(y_pred_all, dtype=float)

    dt = np.diff(np.concatenate([[0.0], t_all]))
    dm = np.diff(np.concatenate([[0.0], m]))
    dt = np.maximum(dt, 1e-8)
    dm = np.maximum(dm, 1e-8)

    lam = dm / dt
    loglik_inc = np.log(lam) - lam * dt
    loglik_cum = np.cumsum(loglik_inc)
    return loglik_cum


#（可选）JM 间隔拟合图，如果你有需要：
def plot_jm_intervals(jm_res):
    k_train = jm_res["k_train"]
    d_train = jm_res["d_train"]
    yhat_train = jm_res["yhat_train"]
    k_valid = jm_res["k_valid"]
    d_valid = jm_res["d_valid"]
    yhat_valid = jm_res["yhat_valid"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(k_train, d_train, s=15, label="Train Δt (truth)", alpha=0.7)
    ax.plot(k_train, yhat_train, label="Train Δt (JM)", linewidth=1.8)

    if len(k_valid) > 0:
        ax.scatter(k_valid, d_valid, s=15, label="Valid Δt (truth)", alpha=0.7)
        ax.plot(k_valid, yhat_valid, label="Valid Δt (JM)",
                linewidth=1.8, linestyle="--")

    ax.set_xlabel("故障序号 k")
    ax.set_ylabel("间隔 Δt_k")
    ax.set_title("JM 模型：间隔域拟合")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return fig
