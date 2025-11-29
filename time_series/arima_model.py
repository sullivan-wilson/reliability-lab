import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

# 忽略 statsmodels 可能产生的收敛警告
warnings.filterwarnings("ignore")


class ArimaReliability:
    def __init__(self, order=(1, 1, 1)):
        """
        :param order: (p, d, q) 参数，默认 (1,1,1)，实际项目中最好由 Auto-ARIMA 确定
        """
        self.order = order
        self.model_fit = None
        self.history = None

    def fit(self, data):
        """
        :param data: 1D 序列，建议使用失效间隔时间(Inter-failure times)或分段时间内的失效数
        """
        self.history = list(data)
        # 简单处理：每次都在全量数据上训练
        # 实际严谨做法可能需要差分检测等
        try:
            model = ARIMA(self.history, order=self.order)
            self.model_fit = model.fit()
        except Exception as e:
            print(f"ARIMA 训练失败: {e}")

    def predict(self, n_steps=1):
        """
        向后预测 n_steps 步
        返回: (拟合的历史数据, 未来的预测数据)
        """
        if self.model_fit is None:
            return np.zeros(len(self.history)), np.zeros(n_steps)

        # 获取历史拟合值
        # predict start=1 表示从第2个点开始预测（因为ARIMA通常涉及差分）
        # 这里为了对齐，我们取 fittedvalues
        fitted_hist = self.model_fit.fittedvalues

        # 预测未来
        forecast_res = self.model_fit.forecast(steps=n_steps)

        return fitted_hist, forecast_res