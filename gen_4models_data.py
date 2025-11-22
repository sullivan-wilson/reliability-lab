# gen_4models_data.py
# 生成一份 4 个模型（GO / JM / MO / S）都适用的模拟数据集

import os
import numpy as np
import pandas as pd

# 固定随机种子，保证可复现实验
np.random.seed(2025)

# 观测故障数 N（数据点数）
N = 80

# JM 模型参数设定：总故障数 N0 远大于 N，避免尾部爆炸
N0 = 120          # 理论总故障数
phi = 0.02        # 故障检测率参数，越大故障越密集

# 第 k 次故障的强度 λ_k = φ (N0 - k + 1)
k = np.arange(1, N + 1, dtype=float)
lambdas = phi * (N0 - k + 1)

# 按强度采样间隔 Δt_k ~ Exp(rate = λ_k)，期望值 1/λ_k 随 k 增大
intervals = np.random.exponential(scale=1.0 / lambdas)

# 四舍五入一下，方便看
intervals = np.round(intervals, 3)

# 累加得到累计时刻 t_k
t = np.round(np.cumsum(intervals), 3)

df = pd.DataFrame({
    "interval": intervals,
    "t": t,
})

# 输出到 data 目录
os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "ntds_4models_80points.csv")
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"Saved {len(df)} points to {out_path}")
print(df.head())
