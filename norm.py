import numpy as np


class BatchNormalization:
    def __init__(self, input_size, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.input_size = input_size
        self.gamma = np.ones((1, input_size))  # 初始化缩放参数
        self.beta = np.zeros((1, input_size))  # 初始化偏移参数
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        self.batch_size = None
        self.x_hat = None

    def forward(self, x, is_training=True):
        self.batch_size, input_size = x.shape

        if is_training:
            # 计算当前批次的均值和方差
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)

            # 标准化
            self.x_hat = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # 更新运行时的均值和方差（滑动平均）
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            # 在推理阶段使用运行时的均值和方差来标准化
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # 缩放和偏移
        out = self.gamma * self.x_hat + self.beta

        return out

    def backward(self, dout):
        # 计算gamma和beta的梯度
        dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        # 计算x_hat的梯度
        dx_hat = dout * self.gamma

        # 计算方差和均值的梯度
        dvar = np.sum(dx_hat * (self.x - self.running_mean) * (-0.5) * np.power(self.running_var + self.epsilon, -1.5),
                      axis=0, keepdims=True)
        dmean = np.sum(dx_hat * (-1 / np.sqrt(self.running_var + self.epsilon)), axis=0,
                       keepdims=True) + dvar * np.mean(-2.0 * (self.x - self.running_mean), axis=0, keepdims=True) * (
                            1.0 / self.x.shape[0])
