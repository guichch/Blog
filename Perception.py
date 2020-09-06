import numpy as np
from sklearn.datasets import load_iris


class Perception(object):
    """
    eta:学习率
    w_:神经分叉权重向量
    error_:用于记录神经元判断出错次数
    """
    def __init__(self, eta=0.01):
        self.eta = eta
        pass

    def fit(self, x, y):
        """
                输入训练数据，培训神经元，x输入样本向量，y对应样本分类

                x:shape[n_samples, n_features]
                x:[[1, 2, 3], [4, 5, 6]]
                n_samples:2
                n_features:3

                y:[1, -1]
        """

        # 初始化权重向量为0
        self.w_ = np.zeros(1 + x.shape[1])

        # 拓展x向量
        x0 = np.ones(x.shape[0])
        x = np.c_[x0, x]

        # 计数
        i = 0

        # 训练
        while i < x.shape[0]:
            for xi, target in zip(x, y):
                i += 1
                if target * np.dot(xi, self.w_) <= 0:
                    self.w_ = self.w_ + self.eta * xi * target
                    break
        return self.w_

    def prediction(self, x):
        return np.sign(np.dot(x, self.w_[1:]) + self.w_[0])

data = load_iris()
x = data['data']
y = data['target']
for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1
    else:
        y[i] = 1
print(y)
example = Perception(1)
w = example.fit(x, y)
print(w)
print(example.prediction(x[49]))










