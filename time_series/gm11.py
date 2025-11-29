import numpy as np
import pandas as pd


class GM11:
    def __init__(self):
        self.a = None  # 发展系数
        self.b = None  # 灰作用量
        self.x0 = None  # 原始序列
        self.preds = None  # 预测结果

    def fit(self, data):
        """
        训练 GM(1,1) 模型
        :param data: 原始数据序列 (1D numpy array or list) - 通常是累计失效数或失效间隔
        """
        self.x0 = np.array(data)
        n = len(self.x0)

        # 1. 一次累加生成 (AGO)
        x1 = np.cumsum(self.x0)

        # 2. 构造紧邻均值生成序列 Z
        z1 = 0.5 * (x1[:-1] + x1[1:])

        # 3. 构造数据矩阵 B 和数据向量 Y
        B = np.vstack([-z1, np.ones(n - 1)]).T
        Y = self.x0[1:]

        # 4. 最小二乘法求解参数 a, b
        # (B.T * B)^-1 * B.T * Y
        try:
            self.a, self.b = np.linalg.inv(B.T @ B) @ B.T @ Y
        except np.linalg.LinAlgError:
            print("GM(1,1) 矩阵奇异，无法求逆，数据可能不适合。")
            self.a, self.b = 0, 0

    def predict(self, n_steps=0):
        """
        :param n_steps: 向后预测多少步
        :return: 包含历史拟合值 + 未来预测值的完整序列
        """
        if self.a is None or self.b is None:
            raise ValueError("模型未训练，请先调用 fit()")

        n = len(self.x0)
        total_len = n + n_steps

        preds_cumulative = []
        preds_original = []

        # 5. 时间响应函数求解 (计算累加预测值)
        # x1(k+1) = (x0(1) - b/a) * e^(-ak) + b/a
        # 注意：这里 k 从 0 开始，对应原数据的 k=1, 2...

        factor = self.x0[0] - self.b / self.a

        for k in range(total_len):
            if k == 0:
                val = self.x0[0]  # 初始值保持一致
            else:
                val = factor * np.exp(-self.a * k) + self.b / self.a
            preds_cumulative.append(val)

        # 6. 累减还原 (IAGO) 得到原始数据预测值
        preds_cumulative = np.array(preds_cumulative)
        preds_original = np.zeros(total_len)
        preds_original[0] = preds_cumulative[0]
        for i in range(1, total_len):
            preds_original[i] = preds_cumulative[i] - preds_cumulative[i - 1]

        return preds_original