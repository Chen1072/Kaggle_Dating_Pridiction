import math
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 24,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],  # SimSun
}

rcParams.update(config)
def ln(x):
    result = np.empty_like(x, dtype=float)  # Create an empty array of the same shape as x
    result[x == 0] = -sys.maxsize          # Set -sys.maxsize where x is 0
    result[x != 0] = np.log(x[x != 0])     # Compute log only on non-zero elements
    return result

plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("Thyroid_Diff.csv")

# 计算分割点
split_point = int(len(df) * 0.99)

# 分割数据集
train_df = df[:split_point]
test_df = df[split_point:]

# 转换为numpy数组，假设最后一列是目标变量
train_np = train_df.to_numpy()
test_np = test_df.to_numpy()

# 定义训练集
t_train = train_np[:, -1].reshape(-1, 1)
X_train = train_np[:, :-1]
N_train = X_train.shape[0]
#X_train = np.hstack([np.ones([N_train, 1]), X_train])  # add bias

# 定义测试集
t_test = test_np[:, -1].reshape(-1, 1)
X_test = test_np[:, :-1]
N_test = X_test.shape[0]
#X_test = np.hstack([np.ones([N_test, 1]), X_test])  # add bias
# Extra Useful Definition
M = X_train.shape[1]
XTX = X_train.T @ X_train
XTt = X_train.T @ t_train


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============= Gradient Descent with Cross-Entropy Loss ================
maxIter = 10000
eta = 1e-5  # learning rate
epsilon = 1e-5  # threshold for stopping criterion
w = np.random.uniform(0, 1, (M, 1))
E = np.zeros(maxIter + 1)

# Initial prediction
y = sigmoid(np.matmul(X_train, w))
# Initial cross-entropy error function value
E[0] = -np.sum(t_train * ln(y) + (1 - t_train) * ln(1 - y))

for k in range(1, maxIter + 1):
    w_prev = w
    # Update gradient: grad = X_train.T @ (y - t_train)
    grad = np.matmul(X_train.T, (y - t_train))
    # Update weights
    w = w - eta * grad
    # Update predictions
    y = sigmoid(np.matmul(X_train, w))
    # Update error function's value (Cross-Entropy)
    E[k] = -np.sum(t_train * ln(y) + (1 - t_train) * ln(1 - y))
    # Check stopping criterion
    if np.linalg.norm(w - w_prev) < epsilon:
        break

print("Stopping criterion reached at iteration:", k)
print("The first 10 elements of w are:\n", w)
MSE_train = np.mean((y - t_train) ** 2)
y_test = sigmoid(np.matmul(X_test, w))
MSE_test = np.mean((y_test - t_test) ** 2)
print("Training Cross-Entropy Loss: ", E[k])
print("Test MSE: ", MSE_test)
print("First 10 elements of predicted outcomes: \n", y_test[:10])

# Plot the error function
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1, k + 1), E[0:k], 'r-', label="Cross-Entropy Loss", linewidth=2)
plt.xlabel('Iteration Number')
plt.ylabel('Cross-Entropy Loss')
plt.title('Error Function Convergence')
plt.grid(True)
plt.show()