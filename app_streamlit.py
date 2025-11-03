# app_streamlit.py
# Streamlit 平台：上传CSV -> 勾选模型 -> 运行 -> 指标&曲线 -> 导出PDF

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils import load_times, cum_to_intervals, ensure_output_dir
from metrics import mae, rmse, mspe, r2, ae, ks_test_residuals
from srgm.go import GOModel
from srgm.jm import JMModel
from srgm.mo import MOModel
from srgm.s_shaped import SShapedModel

# ---------- 页面基础 ----------
st.set_page_config(page_title="软件可靠性增长模型平台", layout="wide")
st.title("软件可靠性增长模型（SRGM）小平台")
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

def fit_and_eval_on_cum(model, t_train, t_valid):
    """GO/MO/S：按累计时刻做极大似然拟合，并算指标/曲线。"""
    t_all = np.concatenate([t_train, t_valid])
    model.fit(t_train)  # 拟合只用训练段

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

def eval_jm_on_intervals(t_train, t_valid):
    """JM：在“间隔域”做严格评估（更符合 JM 定义）。返回指标与可视化数据。"""
    d_train = cum_to_intervals(t_train)
    d_valid = cum_to_intervals(np.concatenate([t_train, t_valid]))[len(d_train):]

    jm = JMModel().fit(d_train)

    k_train = np.arange(1, len(d_train) + 1, dtype=float)
    k_valid = np.arange(len(d_train) + 1, len(d_train) + len(d_valid) + 1, dtype=float)

    # 期望间隔 E[Δt_k] = 1/(φ (N0 - k + 1))
    yhat_train = np.array([jm.expected_interval(int(k)) for k in k_train], dtype=float)
    yhat_valid = np.array([jm.expected_interval(int(k)) for k in k_valid], dtype=float)

    # 指标（在间隔域评估）
    mtrain = {"MAE": mae(d_train, yhat_train), "RMSE": rmse(d_train, yhat_train),
              "MSPE": mspe(d_train, yhat_train), "R2": r2(d_train, yhat_train)}
    mvalid = {"MAE": mae(d_valid, yhat_valid), "RMSE": rmse(d_valid, yhat_valid),
              "MSPE": mspe(d_valid, yhat_valid), "R2": r2(d_valid, yhat_valid),
              "AE_last": ae(d_valid[-1], yhat_valid[-1]) if len(d_valid) > 0 else np.nan}

    return {
        "metrics_train": mtrain, "metrics_valid": mvalid,
        "k_train": k_train, "d_train": d_train, "yhat_train": yhat_train,
        "k_valid": k_valid, "d_valid": d_valid, "yhat_valid": yhat_valid
    }

def build_metrics_table(results: dict) -> pd.DataFrame:
    """把各模型的 train/valid 指标合成一张 DataFrame。"""
    rows = []
    for name, res in results.items():
        mt, mv = res["metrics_train"], res["metrics_valid"]
        rows.append([
            name,
            mt.get("MAE"), mt.get("RMSE"), mt.get("MSPE"), mt.get("R2"),
            mv.get("MAE"), mv.get("RMSE"), mv.get("MSPE"), mv.get("R2"), mv.get("AE_last")
        ])
    return pd.DataFrame(rows, columns=[
        "Model",
        "Train_MAE","Train_RMSE","Train_MSPE","Train_R2",
        "Valid_MAE","Valid_RMSE","Valid_MSPE","Valid_R2","Valid_AE_last"
    ])

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
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 12))

    # 图片
    for img_bytes in images:
        # 控制宽度以适应 A4
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
uploaded = st.sidebar.file_uploader("上传 CSV（含列 't' 或 'interval'）", type=["csv"])
use_sample = st.sidebar.checkbox("没有文件就用示例 data/ntds_sample.csv", value=True)

selected_models = st.sidebar.multiselect(
    "选择模型", ["GO", "JM", "MO", "S"], default=["GO","JM","MO","S"]
)
train_ratio = st.sidebar.slider("训练集比例", min_value=0.5, max_value=0.95, value=0.82, step=0.01)
run_btn = st.sidebar.button("🚀 运行")

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
st.write(f"**数据点数 N = {N}**")
split = int(max(5, min(N-1, round(N*train_ratio))))
t_train, t_valid = t[:split], t[split:]
st.write(f"训练集：{len(t_train)}，验证集：{len(t_valid)}（可在侧边栏调整比例）")

# ---------- 运行 ----------
if run_btn:
    results = {}
    # GO
    if "GO" in selected_models:
        res = fit_and_eval_on_cum(GOModel(), t_train, t_valid)
        results["GO"] = res
    # MO
    if "MO" in selected_models:
        res = fit_and_eval_on_cum(MOModel(), t_train, t_valid)
        results["MO"] = res
    # S
    if "S" in selected_models:
        res = fit_and_eval_on_cum(SShapedModel(), t_train, t_valid)
        results["S"] = res
    # JM（间隔域评估）
    if "JM" in selected_models:
        res = eval_jm_on_intervals(t_train, t_valid)
        results["JM"] = res

    # ---------- 指标表 ----------
    # 注意：JM 的指标在“间隔域”，其他模型在“累计域”。为清晰起见我们统一放一张表并在下面说明。
    metric_df = build_metrics_table(results)
    st.subheader("指标汇总")
    st.dataframe(metric_df, use_container_width=True)
    st.caption("注：JM 为 **间隔域** 指标（与 GO/MO/S 的累计域不同），用来更符合 JM 定义。")

    # ---------- 图表 ----------
    img_bytes_to_export = []

    # 1) 累计失效对比（只画 GO/MO/S）
    if any(m in results for m in ["GO","MO","S"]):
        st.subheader("累计失效曲线（GO/MO/S）")
        t_all = np.concatenate([t_train, t_valid])
        x_idx = np.arange(1, len(t_all)+1, dtype=float)
        fig = plt.figure()
        plt.plot(x_idx, x_idx, label="Truth")  # 真实累计故障 y=x
        for name in ["GO","MO","S"]:
            if name in results:
                yh = np.concatenate([results[name]["y_hat_train"], results[name]["y_hat_valid"]])
                plt.plot(x_idx, yh, label=name)
        plt.xlabel("Failure index")
        plt.ylabel("Cumulative failures (model vs truth)")
        plt.legend()
        st.pyplot(fig)
        img_bytes_to_export.append(fig_to_bytes(fig))

    # 2) JM 间隔域拟合图
    if "JM" in results:
        st.subheader("JM：间隔域拟合（Δt_k vs 期望间隔）")
        res = results["JM"]
        fig2 = plt.figure()
        # 训练
        plt.plot(res["k_train"], res["d_train"], label="Train Δt (truth)")
        plt.plot(res["k_train"], res["yhat_train"], label="Train Δt (JM)")
        # 验证
        if len(res["k_valid"]) > 0:
            plt.plot(res["k_valid"], res["d_valid"], label="Valid Δt (truth)")
            plt.plot(res["k_valid"], res["yhat_valid"], label="Valid Δt (JM)")
        plt.xlabel("Failure index k")
        plt.ylabel("Interval Δt")
        plt.legend()
        st.pyplot(fig2)
        img_bytes_to_export.append(fig_to_bytes(fig2))

    # ---------- 导出 PDF ----------
    st.subheader("导出报告")
    meta = f"数据源: {source_name} | 训练集: {len(t_train)} | 验证集: {len(t_valid)} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    pdf_bytes = make_pdf_bytes("软件可靠性增长模型实验报告", metric_df, img_bytes_to_export, meta)
    st.download_button(
        label="⬇️ 下载 PDF 报告",
        data=pdf_bytes,
        file_name="SRGM_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

else:
    st.info("在左侧选择模型和训练集比例，然后点击 **运行**。")
