# app_streamlit.py
# Streamlit 平台：上传CSV -> 勾选模型 -> 运行 -> 指标&曲线 -> 导出PDF

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import streamlit as st

from utils import load_times, cum_to_intervals, ensure_output_dir
from metrics import mae, rmse, mspe, r2, ae, ks_test_residuals, ks_on_u_values
# 引入原有模型
from srgm.go import GOModel
from srgm.jm import JMModel
from srgm.mo import MOModel
from srgm.s_shaped import SShapedModel
# 引入新模型 (请确保文件路径正确)
from time_series.gm11 import GM11
from time_series.arima_model import ArimaReliability

from ml_models.svr_model import SVRReliability
from ml_models.bpnn_model import BPNNReliability
# 引入可视化辅助
from viz_srgm import (
    build_metrics_table,
    plot_cum_failures,
    compute_u_sequence,
    plot_u_y,
    compute_resid_corr,
    compute_plr_loglik,
    plot_jm_intervals,
)

# 让 matplotlib 支持中文显示
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans", "WenQuanYi Zen Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

# ---------- 页面基础 ----------
st.set_page_config(page_title="软件可靠性增长模型平台", layout="wide")
st.title("软件可靠性增长模型")
st.caption("上传 CSV → 勾选模型 → 一键运行 → 指标 & 曲线 → 导出 PDF 报告")


# ---------- 帮助函数 ----------
def load_csv_t_or_interval(file_obj: io.BytesIO | str):
    """允许 CSV 有 't'(累计时刻) 或 'interval'(间隔) 两种格式。返回累计时刻 t(np.ndarray)。"""
    df = pd.read_csv(file_obj)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if "t" in df.columns:
        t = df["t"].to_numpy(dtype=float)
        if np.any(np.diff(t) <= 0):
            # 允许相等时间发生多次失效，但不允许时间倒流
            if np.any(np.diff(t) < 0):
                raise ValueError("'t' 必须非递减")
        return t
    elif "interval" in df.columns:
        d = df["interval"].to_numpy(dtype=float)
        if np.any(d <= 0):
            # 严格来说间隔应为正，但极短时间间隔允许为0
            pass
        return np.cumsum(d)
    else:
        raise ValueError("CSV 必须包含列 't'（累计时刻）或 'interval'（间隔）")


def ess(y_true, y_pred) -> float:
    """误差平方和 ESS = sum((actual - predicted)^2)。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum((y_true - y_pred) ** 2))


def fit_and_eval_on_cum(model, t_train, t_valid):
    """
    GO/MO/S (SRGM)：在累计时刻域做极大似然拟合。
    预测目标：给定时刻 t，预测累计失效数 m(t)。
    """
    t_all = np.concatenate([t_train, t_valid])
    model.fit(t_train)  # 拟合只用训练段

    # SRGM 是给出时间 t，预测失效数 y
    y_true_train = np.arange(1, len(t_train) + 1, dtype=float)
    y_hat_train = model.predict_cum(t_train)

    y_true_valid = np.arange(len(t_train) + 1, len(t_all) + 1, dtype=float)
    y_hat_valid = model.predict_cum(t_all)[len(t_train):]

    metrics_train = {
        "MAE": mae(y_true_train, y_hat_train),
        "AE": ae(y_true_train, y_hat_train),
        "RMSE": rmse(y_true_train, y_hat_train),
        "MSPE": mspe(y_true_train, y_hat_train),
        "R2": r2(y_true_train, y_hat_train),
        "ESS": ess(y_true_train, y_hat_train),
    }
    metrics_valid = {
        "MAE": mae(y_true_valid, y_hat_valid),
        "AE": ae(y_true_valid, y_hat_valid),
        "RMSE": rmse(y_true_valid, y_hat_valid),
        "MSPE": mspe(y_true_valid, y_hat_valid),
        "R2": r2(y_true_valid, y_hat_valid),
        "ESS": ess(y_true_valid, y_hat_valid),
        "AE_last": ae(y_true_valid[-1], y_hat_valid[-1]) if len(y_hat_valid) > 0 else 0,
    }

    ks_train = ks_test_residuals(y_true_train, y_hat_train)
    ks_valid = ks_test_residuals(y_true_valid, y_hat_valid)

    return {
        "type": "SRGM",  # 标记类型
        "metrics_train": metrics_train,
        "metrics_valid": metrics_valid,
        "ks_train": ks_train,
        "ks_valid": ks_valid,
        "y_true_train": y_true_train,  # 累计失效数 1,2,3...
        "y_hat_train": y_hat_train,
        "y_true_valid": y_true_valid,
        "y_hat_valid": y_hat_valid,
        "model_obj": model  # 存储模型对象以便后续使用
    }


def fit_and_eval_time_series(model_name, t_train, t_valid, arima_order=(1, 1, 1), params=None):
    """
    GM(1,1) / ARIMA：时间序列预测。
    预测目标：给定失效序号 i，预测失效时间 t_i。
    注意：这里的预测方向与 SRGM 相反（SRGM是 t->m(t)，TS是 i->t_i）。
    为了统一画图（画 m(t) 曲线），我们需要把预测出的 t_i 转换回 (t, m(t)) 的形式。
    """
    t_all = np.concatenate([t_train, t_valid])
    # 确保 params 是字典，防止 NoneType 错误
    if params is None:
        params = {}
    # 训练模型：输入是失效时间序列
    if model_name == "GM(1,1)":
        model = GM11()
        model.fit(t_train)
        # 预测：历史拟合 + 未来预测
        # predict 返回的是完整的序列 (len = len(t_train) + len(t_valid))
        preds_all = model.predict(n_steps=len(t_valid))

    elif model_name == "ARIMA":
        model = ArimaReliability(order=arima_order)
        model.fit(t_train)
        hist_fit, future_pred = model.predict(n_steps=len(t_valid))
        preds_all = np.concatenate([hist_fit, future_pred])
    # --- 新增代码 (SVR & BP) ---
    elif model_name == "SVR":
        # params 结构: {'window': 3, 'C': 100, 'gamma': 0.1}
        w = params.get('window', 3)
        c = params.get('C', 100)
        g = params.get('gamma', 0.1)
        model = SVRReliability(window_size=w, C=c, gamma=g)
        model.fit(t_train)
        hist_fit, future_pred = model.predict(n_steps=len(t_valid))
        preds_all = np.concatenate([hist_fit, future_pred])

    elif model_name == "BPNN":
        # params 结构: {'window': 3, 'hidden': (100,), 'iter': 2000}
        w = params.get('window', 3)
        h = params.get('hidden', (100,))
        itr = params.get('iter', 2000)
        model = BPNNReliability(window_size=w, hidden_layer_sizes=h, max_iter=itr)
        model.fit(t_train)
        hist_fit, future_pred = model.predict(n_steps=len(t_valid))
        preds_all = np.concatenate([hist_fit, future_pred])

    # --- 转换回 m(t) 视角进行指标计算 ---
    # 时间序列模型直接预测的是“第i次失效发生的时间”
    # 所以 y_true 是 t_train/t_valid (时间)
    # y_pred 是模型输出的预测时间

    # 训练部分
    pred_t_train = preds_all[:len(t_train)]
    # 验证部分
    pred_t_valid = preds_all[len(t_train):]

    # 为了能在“指标总览”里和 SRGM 比较，我们通常比较“时间误差”或者“失效数误差”。
    # SRGM 计算的是失效数误差 (预测 m(t) vs 真实 i)。
    # TS 模型计算的是时间误差 (预测 t_i vs 真实 t)。
    # 这里为了展示 TS 模型的原生性能，我们计算 **时间误差**。
    # 并在表格备注中说明。

    metrics_train = {
        "MAE": mae(t_train, pred_t_train),
        "RMSE": rmse(t_train, pred_t_train),
        "MSPE": mspe(t_train, pred_t_train),  # 时间的百分比误差
        "R2": r2(t_train, pred_t_train),
        "ESS": ess(t_train, pred_t_train),
        "AE": ae(t_train, pred_t_train)
    }

    metrics_valid = {
        "MAE": mae(t_valid, pred_t_valid),
        "RMSE": rmse(t_valid, pred_t_valid),
        "MSPE": mspe(t_valid, pred_t_valid),
        "R2": r2(t_valid, pred_t_valid),
        "ESS": ess(t_valid, pred_t_valid),
        "AE_last": ae(t_valid[-1], pred_t_valid[-1]) if len(t_valid) > 0 else 0
    }

    return {
        "type": "TimeSeries",
        "metrics_train": metrics_train,
        "metrics_valid": metrics_valid,
        # 用于画图的数据：
        # x轴是 时间(预测值), y轴是 失效序号(1,2,3...)
        "pred_t_all": preds_all,
        "t_train_true": t_train,
        "t_valid_true": t_valid
    }


def eval_jm_on_intervals(t_train, t_valid):
    """
    JM：在“间隔域”做评估。
    """
    d_train = cum_to_intervals(t_train)
    d_valid = cum_to_intervals(np.concatenate([t_train, t_valid]))[len(d_train):]

    jm = JMModel().fit(d_train)

    k_train = np.arange(1, len(d_train) + 1, dtype=float)
    k_valid = np.arange(len(d_train) + 1, len(d_train) + len(d_valid) + 1, dtype=float)

    # 期望间隔
    yhat_train = np.array([jm.expected_interval(int(k)) for k in k_train], dtype=float)
    yhat_valid = np.array([jm.expected_interval(int(k)) for k in k_valid], dtype=float)

    # 指标（间隔域）
    mtrain = {
        "MAE": mae(d_train, yhat_train),
        "AE": ae(d_train, yhat_train),
        "RMSE": rmse(d_train, yhat_train),
        "MSPE": mspe(d_train, yhat_train),
        "R2": r2(d_train, yhat_train),
        "ESS": ess(d_train, yhat_train),
    }
    mvalid = {
        "MAE": mae(d_valid, yhat_valid),
        "AE": ae(d_valid, yhat_valid),
        "RMSE": rmse(d_valid, yhat_valid),
        "MSPE": mspe(d_valid, yhat_valid),
        "R2": r2(d_valid, yhat_valid),
        "ESS": ess(d_valid, yhat_valid),
        "AE_last": ae(d_valid[-1], yhat_valid[-1]) if len(d_valid) > 0 else np.nan,
    }

    return {
        "type": "JM",
        "metrics_train": mtrain,
        "metrics_valid": mvalid,
        "k_train": k_train,
        "d_train": d_train,
        "yhat_train": yhat_train,
        "k_valid": k_valid,
        "d_valid": d_valid,
        "yhat_valid": yhat_valid,
    }


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def make_pdf_bytes(title: str, metrics_df: pd.DataFrame, images: list[bytes], meta_text: str) -> bytes:
    """用 reportlab 生成 PDF 并返回字节流。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []

    # 支持中文的字体设置（ReportLab 默认不支持中文，这里做简单回退处理）
    # 如果生产环境需要中文PDF，需要注册中文字体。这里简化为英文标题或提示。
    elems.append(Paragraph(title, styles["Title"]))
    elems.append(Paragraph(meta_text, styles["Normal"]))
    elems.append(Spacer(1, 8))

    # 指标表
    # 将 DataFrame 转为列表
    data = [metrics_df.columns.tolist()] + metrics_df.round(4).astype(str).values.tolist()

    # 自动计算列宽（简单策略）
    col_widths = [80] + [50] * (len(metrics_df.columns) - 1)

    tbl = Table(data, repeatRows=1, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 12))

    # 图片
    for img_bytes in images:
        w = 460  # px
        h = 300
        elems.append(Image(io.BytesIO(img_bytes), width=w, height=h))
        elems.append(Spacer(1, 12))

    doc.build(elems)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# ---------- 侧边栏：参数 ----------
st.sidebar.header("参数设置")
uploaded = st.sidebar.file_uploader("上传 CSV（含列 't' 或 'interval'）", type=["csv"])
use_sample = st.sidebar.checkbox("使用示例 data/ntds_sample.csv", value=True)

# 模型选择
st.sidebar.subheader("选择模型")
selected_models = st.sidebar.multiselect(
    "SRGM 模型", ["GO", "JM", "MO", "S"], default=["GO", "JM"]
)
selected_ts_models = st.sidebar.multiselect(
    "时间序列/智能模型", ["GM(1,1)", "ARIMA", "SVR", "BPNN"], default=["GM(1,1)"]
)

# ARIMA 参数 (仅当选择了ARIMA时显示)
if "ARIMA" in selected_ts_models:
    st.sidebar.caption("ARIMA 参数 (p,d,q)")
    c1, c2, c3 = st.sidebar.columns(3)
    p_val = c1.number_input("p", 0, 5, 1)
    d_val = c2.number_input("d", 0, 2, 1)  # 累积时间非平稳，d>=1
    q_val = c3.number_input("q", 0, 5, 1)
    arima_order = (p_val, d_val, q_val)
else:
    arima_order = (1, 1, 1)

# SVR 参数 (新增)
svr_params = {}
if "SVR" in selected_ts_models:
    with st.sidebar.expander("SVR 参数 (智能算法)"):
        svr_win = st.slider("滑动窗口 (Look-back)", 2, 10, 3, key="svr_w")
        svr_c = st.number_input("C (正则化)", 1.0, 1000.0, 100.0, step=10.0, key="svr_c")
        svr_g = st.number_input("Gamma", 0.001, 1.0, 0.1, step=0.01, key="svr_g")
        svr_params = {'window': svr_win, 'C': svr_c, 'gamma': svr_g}

# BPNN 参数 (新增)
bp_params = {}
if "BPNN" in selected_ts_models:
    with st.sidebar.expander("BP 神经网络参数"):
        bp_win = st.slider("滑动窗口", 2, 10, 3, key="bp_w")
        bp_node = st.number_input("隐藏层节点数", 10, 500, 100, step=10, key="bp_n")
        bp_iter = st.number_input("最大迭代次数", 500, 5000, 2000, step=100, key="bp_i")
        bp_params = {'window': bp_win, 'hidden': (bp_node,), 'iter': bp_iter}


train_ratio = st.sidebar.slider("训练集比例", min_value=0.5, max_value=0.95, value=0.82, step=0.01)
run_btn = st.sidebar.button("运行分析", type="primary")

# ---------- 主体：数据加载 ----------
t = None
source_name = None

try:
    if uploaded is not None:
        t = load_csv_t_or_interval(uploaded)
        source_name = uploaded.name
    elif use_sample and os.path.exists("data/ntds_sample.csv"):
        t = load_csv_t_or_interval("data/ntds_sample.csv")
        source_name = "data/ntds_sample.csv"
    elif os.path.exists("data/ntds_from_slide.csv"):
        t = load_csv_t_or_interval("data/ntds_from_slide.csv")
        source_name = "data/ntds_from_slide.csv"
except Exception as e:
    st.error(f"读取 CSV 出错：{e}")

if t is None:
    st.info("请在左侧上传 CSV（包含列 't' 或 'interval'），或勾选使用项目自带示例。")
    st.stop()

N = len(t)
st.write(f"**数据概览：** 总数据点数 N = {N}，来源：{source_name}")
split = int(max(5, min(N - 1, round(N * train_ratio))))
t_train, t_valid = t[:split], t[split:]
st.write(f"训练集：{len(t_train)} 个点 (前 {train_ratio * 100:.0f}%) | 验证集：{len(t_valid)} 个点")

# ---------- 运行 / 结果缓存 ----------
if run_btn:
    results: dict[str, dict] = {}

    # 1. 运行 SRGM 模型
    if "GO" in selected_models:
        results["GO"] = fit_and_eval_on_cum(GOModel(), t_train, t_valid)
    if "MO" in selected_models:
        results["MO"] = fit_and_eval_on_cum(MOModel(), t_train, t_valid)
    if "S" in selected_models:
        results["S"] = fit_and_eval_on_cum(SShapedModel(), t_train, t_valid)
    if "JM" in selected_models:
        results["JM"] = eval_jm_on_intervals(t_train, t_valid)
    if "SVR" in selected_ts_models:
        results["SVR"] = fit_and_eval_time_series("SVR", t_train, t_valid, params=svr_params)

    if "BPNN" in selected_ts_models:
        results["BPNN"] = fit_and_eval_time_series("BPNN", t_train, t_valid, params=bp_params)

    # 2. 运行 时间序列 模型
    if "GM(1,1)" in selected_ts_models:
        results["GM(1,1)"] = fit_and_eval_time_series("GM(1,1)", t_train, t_valid)
    if "ARIMA" in selected_ts_models:
        results["ARIMA"] = fit_and_eval_time_series("ARIMA", t_train, t_valid, arima_order)

    if not results:
        st.warning("请至少选择一个模型。")
        st.stop()

    # 存入 session
    st.session_state["results"] = results
    st.session_state["t_train"] = t_train
    st.session_state["t_valid"] = t_valid
    st.session_state["source_name"] = source_name

# 检查是否有结果
if "results" not in st.session_state:
    st.info("👈 请在左侧选择模型，然后点击 **运行分析**。")
    st.stop()

# 取出结果
results = st.session_state["results"]
t_train = st.session_state["t_train"]
t_valid = st.session_state["t_valid"]
source_name = st.session_state["source_name"]

t_all = np.concatenate([t_train, t_valid])
n_all = len(t_all)
img_bytes_to_export: list[bytes] = []

# ---------- 展示区域 ----------
st.divider()
tab_metrics, tab_curve, tab_diag, tab_jm = st.tabs(
    ["📊 指标总览", "📈 累计失效曲线 m(t)", "🔍 诊断工具(SRGM)", "⏱ JM & TS 拟合"]
)

# 计算指标表
metric_df = build_metrics_table(results)

# ------------ ① 指标总览 ------------
with tab_metrics:
    st.markdown("### 模型性能指标对比")
    st.markdown("""
    > **注意指标的物理意义不同：**
    > * **SRGM (GO, MO, S)**: 预测目标是 **失效数**。指标反映预测失效数的准确度。
    > * **TS (GM(1,1), ARIMA)**: 预测目标是 **时间**。指标反映预测失效时间的准确度。
    > * **JM**: 预测目标是 **间隔**。
    """)
    st.dataframe(metric_df.style.highlight_min(axis=0, color='#d1e7dd'), use_container_width=True)

# ------------ ② 拟合曲线 m(t) ------------
with tab_curve:
    st.markdown("### 累计失效预测曲线 m(t)")
    st.caption("横轴：时间 t，纵轴：累计失效数 m(t)。SRGM 直接输出曲线；TS 模型(GM/ARIMA)通过预测的时间点反推曲线。")

    # 我们需要自定义一个绘图函数来同时支持 SRGM 和 TS 模型的绘制
    fig_cum, ax = plt.subplots(figsize=(10, 6))

    # 1. 画真实数据
    # 真实数据点 (t, m(t)) -> (t_all[i], i+1)
    ax.step(t_all, np.arange(1, n_all + 1), where='post', label="Observed (真实数据)", color='black', linewidth=1.5)

    # 2. 画分割线
    ax.axvline(x=t_train[-1], color='green', linestyle=':', label='Train/Test Split')

    # 3. 遍历所有模型结果并绘制
    colors_cycle = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'purple']
    c_idx = 0

    for name, res in results.items():
        color = colors_cycle[c_idx % len(colors_cycle)]
        c_idx += 1

        if res["type"] == "SRGM":
            # SRGM 结果: x=t_all, y=predict_cum(t_all)
            # 为了平滑，生成更多点
            t_plot = np.linspace(0, t_all[-1] * 1.1, 200)
            model = res["model_obj"]
            y_plot = model.predict_cum(t_plot)
            ax.plot(t_plot, y_plot, linestyle='--', label=f"{name} (SRGM)", color=color)

        elif res["type"] == "TimeSeries":
            # TS 结果: res["pred_t_all"] 是预测的时间点序列 t_1, t_2...
            # 对应的 y 是 1, 2, ...
            pred_times = res["pred_t_all"]
            # 过滤掉非物理意义的时间（比如负数）
            valid_mask = pred_times > 0
            pred_times = pred_times[valid_mask]
            pred_counts = np.arange(1, len(pred_times) + 1)

            # 绘制点图或连线
            ax.plot(pred_times, pred_counts, marker='x', linestyle='--', markersize=4,
                    label=f"{name} (Time-Series)", color=color, alpha=0.7)

        elif res["type"] == "JM":
            # JM 的 m(t) 计算比较复杂（它是分段的），这里通常在 ④ tab 单独看间隔
            # 或者你可以调用 model.expected_failures(t) 如果实现了的话
            pass

    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Cumulative Failures m(t)")
    ax.set_title("Reliability Growth Curves Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig_cum, use_container_width=True)
    img_bytes_to_export.append(fig_to_bytes(fig_cum))

# ------------ ③ 预测有效性诊断 (SRGM Only) ------------
with tab_diag:
    st.markdown("### SRGM 模型诊断 (1.2节)")

    srgm_models = [k for k, v in results.items() if v["type"] == "SRGM"]

    if not srgm_models:
        st.info("当前未选择 SRGM 类模型（GO/MO/S），无法显示此类诊断图。")
    else:
        model_name = st.selectbox("选择要诊断的模型：", srgm_models)
        res = results[model_name]

        # 准备数据
        y_pred_all = np.concatenate([res["y_hat_train"], res["y_hat_valid"]])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**U 图 (U-Plot)**")
            u_all = compute_u_sequence(res)
            if u_all is not None:
                fig_u, _ = plot_u_y(u_all, title_prefix=model_name)
                st.pyplot(fig_u, use_container_width=True)
                img_bytes_to_export.append(fig_to_bytes(fig_u))

        with c2:
            st.markdown("**PLR (序列似然比)**")
            loglik_cum = compute_plr_loglik(t_all, y_pred_all)
            fig_plr, ax_plr = plt.subplots(figsize=(6, 4))
            ax_plr.plot(np.arange(1, len(loglik_cum) + 1), loglik_cum)
            ax_plr.set_title(f"PLR: {model_name}")
            ax_plr.grid(True, alpha=0.3)
            st.pyplot(fig_plr, use_container_width=True)

# ------------ ④ JM & TS 拟合细节 ------------
with tab_jm:
    st.markdown("### 间隔域 & 时间域 拟合详情")

    # 1. JM
    if "JM" in results:
        st.markdown("#### JM 模型：失效间隔拟合")
        fig_jm = plot_jm_intervals(results["JM"])
        st.pyplot(fig_jm, use_container_width=True)
        img_bytes_to_export.append(fig_to_bytes(fig_jm))

    # 2. GM(1,1) / ARIMA
    ts_results = [res for name, res in results.items() if res["type"] == "TimeSeries"]
    if ts_results:
        st.markdown("#### 时间序列模型：失效时间点预测")
        for name, res in results.items():
            if res["type"] != "TimeSeries": continue

            fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
            # 真实时间点
            indices = np.arange(1, len(t_all) + 1)
            ax_ts.plot(indices, t_all, 'k.-', label='True Time')
            # 预测时间点
            pred_t = res["pred_t_all"]
            ax_ts.plot(indices[:len(pred_t)], pred_t, 'r--', label=f'{name} Predicted Time')

            # 分割线
            ax_ts.axvline(x=len(t_train), color='g', linestyle=':', label='Split')

            ax_ts.set_ylabel("Failure Time (t)")
            ax_ts.set_xlabel("Failure Number (i)")
            ax_ts.set_title(f"{name} Prediction Performance")
            ax_ts.legend()
            ax_ts.grid(True, alpha=0.3)
            st.pyplot(fig_ts, use_container_width=True)
            img_bytes_to_export.append(fig_to_bytes(fig_ts))

# ---------- 导出 PDF ----------
st.divider()
col_pdf, _ = st.columns([1, 4])
with col_pdf:
    meta = (
        f"Data: {source_name} | Train/Total: {len(t_train)}/{N} | "
        f"Date: {datetime.now().strftime('%Y-%m-%d')}"
    )
    # 注意：如果环境中没有中文字体，生成的 PDF 中文可能会乱码。
    # 这里 title 用英文以保安全。
    pdf_bytes = make_pdf_bytes(
        "Software Reliability Analysis Report", metric_df, img_bytes_to_export, meta
    )
    st.download_button(
        label="📄 下载 PDF 报告",
        data=pdf_bytes,
        file_name="SRGM_Analysis_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )