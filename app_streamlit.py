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
from srgm.go import GOModel
from srgm.jm import JMModel
from srgm.mo import MOModel
from srgm.s_shaped import SShapedModel
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
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
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
            raise ValueError("'t' 必须严格递增")
        return t
    elif "interval" in df.columns:
        d = df["interval"].to_numpy(dtype=float)
        if np.any(d <= 0):
            raise ValueError("'interval' 必须为正值")
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
    GO/MO/S：在累计时刻域做极大似然拟合，并计算：
    MAE / AE / RMSE / MSPE / R2 / ESS + KS(残差)。
    """
    t_all = np.concatenate([t_train, t_valid])
    model.fit(t_train)  # 拟合只用训练段

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
        "AE_last": ae(y_true_valid[-1], y_hat_valid[-1]),
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
        "y_hat_valid": y_hat_valid,
    }


def eval_jm_on_intervals(t_train, t_valid):
    """
    JM：在“间隔域”做评估，更符合 JM 定义。
    返回间隔域的 MAE / AE / RMSE / MSPE / R2 / ESS 等指标，以及画图所需数据。
    """
    d_train = cum_to_intervals(t_train)
    d_valid = cum_to_intervals(np.concatenate([t_train, t_valid]))[len(d_train):]

    jm = JMModel().fit(d_train)

    k_train = np.arange(1, len(d_train) + 1, dtype=float)
    k_valid = np.arange(len(d_train) + 1, len(d_train) + len(d_valid) + 1, dtype=float)

    # 期望间隔 E[Δt_k] = 1/(φ (N0 - k + 1))
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
    elems.append(Paragraph(title, styles["Title"]))
    elems.append(Paragraph(meta_text, styles["Normal"]))
    elems.append(Spacer(1, 8))

    # 指标表
    data = [metrics_df.columns.tolist()] + metrics_df.round(4).astype(object).values.tolist()
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 12))

    # 图片
    for img_bytes in images:
        w = 480  # px
        h = 320
        elems.append(Image(io.BytesIO(img_bytes), width=w, height=h))
        elems.append(Spacer(1, 12))

    doc.build(elems)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# ---------- 侧边栏：参数 ----------
st.sidebar.header("参数")
uploaded = st.sidebar.file_uploader("上传 CSV（含列 't' 、 'interval'）", type=["csv"])
use_sample = st.sidebar.checkbox("使用示例 data/ntds_sample.csv", value=True)

selected_models = st.sidebar.multiselect(
    "选择模型", ["GO", "JM", "MO", "S"], default=["GO", "JM", "MO", "S"]
)
train_ratio = st.sidebar.slider("训练集比例", min_value=0.5, max_value=0.95, value=0.82, step=0.01)
run_btn = st.sidebar.button("运行")


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
st.write(f"**数据点数 N = {N}（默认34个）**")
split = int(max(5, min(N - 1, round(N * train_ratio))))
t_train, t_valid = t[:split], t[split:]
st.write(f"训练集：{len(t_train)}，验证集：{len(t_valid)}（可在侧边栏调整比例）")


# ---------- 运行 / 结果缓存 ----------
if run_btn:
    # 1）先根据当前侧边栏参数重新计算一次结果
    results: dict[str, dict] = {}

    # GO
    if "GO" in selected_models:
        results["GO"] = fit_and_eval_on_cum(GOModel(), t_train, t_valid)
    # MO
    if "MO" in selected_models:
        results["MO"] = fit_and_eval_on_cum(MOModel(), t_train, t_valid)
    # S
    if "S" in selected_models:
        results["S"] = fit_and_eval_on_cum(SShapedModel(), t_train, t_valid)
    # JM（间隔域评估）
    if "JM" in selected_models:
        results["JM"] = eval_jm_on_intervals(t_train, t_valid)

    if not results:
        st.warning("请至少选择一个模型。")
        st.stop()

    # 2）把结果和当前数据集信息存到 session_state 里，方便之后切换下拉框重跑时复用
    st.session_state["results"] = results
    st.session_state["t_train"] = t_train
    st.session_state["t_valid"] = t_valid
    st.session_state["source_name"] = source_name

# 3）如果从来没按过“运行”，就只提示，不往下画图
if "results" not in st.session_state:
    st.info("在左侧选择模型和训练集比例，然后点击 **运行**。")
    st.stop()

# 4）从 session_state 里取出最新一次的结果
results: dict[str, dict] = st.session_state["results"]
t_train = st.session_state["t_train"]
t_valid = st.session_state["t_valid"]
source_name = st.session_state["source_name"]

# 后面可视化要用的一些公共变量
t_all = np.concatenate([t_train, t_valid])
n_all = len(t_all)
y_true_all = np.arange(1, n_all + 1, dtype=float)

img_bytes_to_export: list[bytes] = []

st.subheader("模型比较与诊断（对应：最小相关误差 / 拟合优度 / PLR / U 图 / Y 图）")

tab_metrics, tab_curve, tab_diag, tab_jm = st.tabs(
    ["① 指标总览", "② 拟合曲线 m(t)", "③ 预测有效性诊断", "④ JM 间隔拟合"]
)

# 统一先算好指标表，后面 tab 和 PDF 都复用这一个 DataFrame
metric_df = build_metrics_table(results)

# ------------ ① 指标总览 ------------
with tab_metrics:
    st.markdown("**1）误差与拟合优度指标**")
    st.dataframe(metric_df, use_container_width=True)
    st.caption(
        "MAE / AE / RMSE / MSPE / R² / ESS 对应课程设计 2.1 页“软件可靠性模型拟合或预测性能优劣指标”。"
        "其中 JM 的指标是在“间隔域”上计算，其余模型在“累计域”上计算。"
    )

# ------------ ② 拟合曲线 m(t) ------------
with tab_curve:
    st.markdown("**2）累计失效曲线 m(t)（GO / MO / S）**")
    fig_cum = plot_cum_failures(results, t_train, t_valid)
    if fig_cum is not None:
        st.pyplot(fig_cum, use_container_width=True)
        img_bytes_to_export.append(fig_to_bytes(fig_cum))

# ------------ ③ 预测有效性诊断：五种方法 ------------
with tab_diag:
    st.markdown("**3）模型预测有效性的五种检验方法（1.2 节）**")

    diag_models = [name for name in results.keys() if name != "JM"]
    if not diag_models:
        st.info("当前只有 JM（间隔域）模型，暂不使用 1.2 节的五种诊断方法。")
    else:
        model_name = st.selectbox("选择要诊断的模型：", diag_models, index=0)
        res = results[model_name]

        y_pred_all = np.concatenate([res["y_hat_train"], res["y_hat_valid"]])

        sub_corr, sub_gof, sub_plr, sub_u, sub_y = st.tabs(
            ["最小相关误差", "拟合优度检验", "序列似然比（PLR）法", "U 图法", "Y 图法"]
        )

        # --- 最小相关误差：残差一阶相关系数 ---
        with sub_corr:
            rho = compute_resid_corr(y_true_all, y_pred_all)
            st.write(f"**残差一阶相关系数 ρ = {rho:.4f}**")
            st.caption(
                "ρ 越接近 0，说明残差越不相关，模型越好；可比较不同模型的 |ρ| 作为“最小相关误差”判断依据。"
            )

        # --- 拟合优度检验：基于 U 序列的 KS 检验 ---
        with sub_gof:
            u_all = compute_u_sequence(res)
            if u_all is None:
                st.info("该模型无法构造 U 序列。")
            else:
                ks_res = ks_on_u_values(u_all)
                st.write(
                    f"**K-S 拟合优度检验：** D = {ks_res['stat']:.4f}, "
                    f"p-value = {ks_res['pvalue']:.4f}"
                )
                st.caption(
                    "原假设：U 值服从 U(0,1) 分布，表示模型给出的故障时间分布与实际一致；"
                    "p 值越大，越难拒绝原假设，即拟合优度越好。"
                )

        # --- 序列似然比（PLR）法：累计 log-likelihood 曲线 ---
        with sub_plr:
            loglik_cum = compute_plr_loglik(t_all, y_pred_all)
            x_idx = np.arange(1, len(loglik_cum) + 1)

            fig_plr, ax_plr = plt.subplots(figsize=(6, 4))
            ax_plr.plot(x_idx, loglik_cum, "-o", markersize=3)
            ax_plr.set_xlabel("故障序号 k")
            ax_plr.set_ylabel("累计 log-likelihood")
            ax_plr.set_title(f"{model_name} 模型的 PLR 序列（累计对数似然）")
            ax_plr.grid(alpha=0.3, linestyle="--", linewidth=0.5)
            fig_plr.tight_layout()
            st.pyplot(fig_plr, use_container_width=True)
            img_bytes_to_export.append(fig_to_bytes(fig_plr))

            st.caption(
                "在相同数据下，不同模型的累计 log-likelihood 曲线可以比较其相对优劣："
                "曲线整体更高（对数似然更大）的模型通常拟合更好。"
            )

        # --- U 图法 ---
        with sub_u:
            u_all = compute_u_sequence(res)
            if u_all is None:
                st.info("该模型无法构造 U 序列。")
            else:
                fig_u, _ = plot_u_y(u_all, title_prefix=model_name)
                st.pyplot(fig_u, use_container_width=True)
                img_bytes_to_export.append(fig_to_bytes(fig_u))
                st.caption(
                    "U 图用样本分位与理论 U(0,1) 分布进行对比。点越靠近对角线，"
                    "说明模型假设的过程（如 NHPP）与观测数据越吻合。"
                )

        # --- Y 图法 ---
        with sub_y:
            u_all = compute_u_sequence(res)
            if u_all is None:
                st.info("该模型无法构造 U 序列。")
            else:
                _, fig_y = plot_u_y(u_all, title_prefix=model_name)
                st.pyplot(fig_y, use_container_width=True)
                st.caption(
                    "Y 图通过 -ln(1-U) 变换，把 U(0,1) 映射到指数分布，"
                    "再与 45° 线进行比较，用于放大尾端差异，进一步检验预测有效性。"
                )

# ------------ ④ JM 间隔拟合图 ------------
with tab_jm:
    st.markdown("**4）JM 间隔域拟合（Δt_k vs 期望间隔）**")
    if "JM" in results:
        fig_jm = plot_jm_intervals(results["JM"])
        st.pyplot(fig_jm, use_container_width=True)
        img_bytes_to_export.append(fig_to_bytes(fig_jm))
        st.caption(
            "这一页可以用来分析 JM 在“故障接近 N₀ 时间隔变长”的特点，与课件中对 JM 假设与局限性的讨论呼应。"
        )
    else:
        st.info("未选择 JM 模型。")

# ---------- 导出 PDF ----------
st.subheader("导出报告")
meta = (
    f"数据源: {source_name} | 训练集: {len(t_train)} | 验证集: {len(t_valid)} | "
    f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
pdf_bytes = make_pdf_bytes(
    "软件可靠性增长模型实验报告", metric_df, img_bytes_to_export, meta
)
st.download_button(
    label="下载 PDF 报告",
    data=pdf_bytes,
    file_name="SRGM_Report.pdf",
    mime="application/pdf",
    use_container_width=True,
)
