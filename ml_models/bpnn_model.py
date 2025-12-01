import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


class BPNNReliability:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=2000, window_size=3):
        """
        :param hidden_layer_sizes: 隐藏层神经元数量，例如 (50, 50) 表示两层，每层50个
        :param window_size: 滑动窗口大小
        """
        self.window_size = window_size
        # MLPRegressor 即多层感知机回归，使用 BP 算法训练
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=42  # 固定随机种子以便复现
        )
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.x_train_raw = None

    def create_dataset(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - self.window_size):
            a = dataset[i:(i + self.window_size)]
            dataX.append(a)
            dataY.append(dataset[i + self.window_size])
        return np.array(dataX), np.array(dataY)

    def fit(self, train_data):
        self.x_train_raw = np.array(train_data).reshape(-1, 1)
        self.train_scaled = self.scaler.fit_transform(self.x_train_raw).flatten()

        if len(self.train_scaled) <= self.window_size:
            raise ValueError("训练数据不足以构建滑动窗口")

        X, y = self.create_dataset(self.train_scaled)
        self.model.fit(X, y)

    def predict(self, n_steps):
        # 逻辑与 SVR 完全一致：递归预测
        current_window = self.train_scaled[-self.window_size:].tolist()
        preds_scaled = []

        for _ in range(n_steps):
            input_seq = np.array(current_window[-self.window_size:]).reshape(1, -1)
            pred_val = self.model.predict(input_seq)[0]
            preds_scaled.append(pred_val)
            current_window.append(pred_val)

        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        preds_original = self.scaler.inverse_transform(preds_scaled).flatten()

        X_train, _ = self.create_dataset(self.train_scaled)
        train_fit_scaled = self.model.predict(X_train)
        train_fit = self.scaler.inverse_transform(train_fit_scaled.reshape(-1, 1)).flatten()

        padding = self.x_train_raw[:self.window_size].flatten()
        history_fit = np.concatenate([padding, train_fit])

        return history_fit, preds_original