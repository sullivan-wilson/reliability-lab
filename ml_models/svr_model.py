import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler


class SVRReliability:
    def __init__(self, kernel='rbf', C=100, gamma=0.1, epsilon=0.1, window_size=3):
        """
        :param window_size: 滑动窗口大小（输入维度），即用过去多少个点预测下一个点
        """
        self.window_size = window_size
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.x_train_raw = None

    def create_dataset(self, dataset):
        """将时间序列转换为监督学习样本 (X, y)"""
        dataX, dataY = [], []
        for i in range(len(dataset) - self.window_size):
            a = dataset[i:(i + self.window_size)]
            dataX.append(a)
            dataY.append(dataset[i + self.window_size])
        return np.array(dataX), np.array(dataY)

    def fit(self, train_data):
        """
        :param train_data: 训练集时间序列 (1D array)
        """
        self.x_train_raw = np.array(train_data).reshape(-1, 1)

        # 1. 数据归一化 (SVR 对尺度非常敏感)
        self.train_scaled = self.scaler.fit_transform(self.x_train_raw).flatten()

        # 2. 构建滑动窗口数据集
        if len(self.train_scaled) <= self.window_size:
            raise ValueError(f"训练数据太少，无法构建窗口。数据点数: {len(self.train_scaled)}, 窗口: {self.window_size}")

        X, y = self.create_dataset(self.train_scaled)

        # 3. 训练模型
        self.model.fit(X, y)

    def predict(self, n_steps):
        """
        递归预测未来 n_steps 步
        """
        # 从训练集末尾提取初始窗口
        current_window = self.train_scaled[-self.window_size:].tolist()
        preds_scaled = []

        # 递归预测：用预测值作为下一步的输入
        for _ in range(n_steps):
            input_seq = np.array(current_window[-self.window_size:]).reshape(1, -1)
            pred_val = self.model.predict(input_seq)[0]
            preds_scaled.append(pred_val)
            current_window.append(pred_val)

        # 反归一化
        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        preds_original = self.scaler.inverse_transform(preds_scaled).flatten()

        # 获取训练集的拟合值 (为了画图完整)
        # 注意：前 window_size 个点无法预测
        X_train, _ = self.create_dataset(self.train_scaled)
        train_fit_scaled = self.model.predict(X_train)
        train_fit = self.scaler.inverse_transform(train_fit_scaled.reshape(-1, 1)).flatten()

        # 补齐前几个点的空缺(用原始值代替或填0)
        padding = self.x_train_raw[:self.window_size].flatten()
        history_fit = np.concatenate([padding, train_fit])

        return history_fit, preds_original