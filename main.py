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

plt.rcParams['axes.unicode_minus'] = False

df_train = pd.read_csv("crime-train.csv")
df_train_np = pd.DataFrame(df_train).to_numpy()
df_test = pd.read_csv("crime-test.csv")
df_test_np = pd.DataFrame(df_test).to_numpy()

# Define t, X, N in train set and test set
t_train = df_train_np[:, :1].reshape(-1, 1)
X_train = df_train_np[:, 1:]
N_train = X_train.shape[0]
X_train = np.hstack([np.ones([N_train, 1]), X_train])  # add bias

t_test = df_test_np[:, :1].reshape(-1, 1)
X_test = df_test_np[:, 1:]
N_test = X_test.shape[0]
X_test = np.hstack([np.ones([N_test, 1]), X_test])  # add bias

# Extra Useful Definition
M = X_train.shape[1]
XTX = X_train.T @ X_train
XTt = X_train.T @ t_train
# ============= Gradient Descent ================
maxIter = 10000
eta = 1e-5 # learning rate
epsilon = 1e-5 # threshold
w = np.random.uniform(0, 1, (M, 1))
E = np.zeros(maxIter + 1)
# Initial prediction
y = np.matmul(X_train, w)
# Initial error function value
E[0] = np.linalg.norm(y - t_train) ** 2 / 2

for k in range(1, maxIter + 1):
    w_prev = w
    # update gradient
    grad = np.matmul(XTX, w) - XTt
    # update weights
    w = w - eta * grad
    # update predictions
    y = np.matmul(X_train, w)
    # update error function's value
    E[k] = np.linalg.norm(y - t_train) ** 2 / 2
    # check stopping criterion
    if max(np.abs(w - w_prev)) < epsilon:
        break
print("The Stopping critirition:", k)
print("The first 10 elements of w are: \n", w[0:10])
MSE_train = np.matmul(np.transpose(y - t_train), (y - t_train)) / N_train
y_test = np.matmul(X_test, w)
MSE_test = np.matmul(np.transpose(y_test - t_test), (y_test - t_test)) / N_test
print("Training MSE: ", MSE_train)
print("Test MSE: ", MSE_test)
print("First 10 elements of predicted crime rate: \n", y_test[0:10])
print("The objective function E converges at:", E[k])

# Least Squares (closed-form solution)
w_star = np.matmul(np.linalg.inv(XTX), XTt)
print("The first 10 elements of closed-form solution w_star are \n:", w_star[0:10])

fig, ax = plt.subplots()
ax.plot(range(1, k + 1), E[0:k], 'r-', label="Error Function", linewidth=2)
plt.xlabel('Iteration Number')
plt.ylabel('Error Function', color="blue")
plt.grid(True)
plt.show()
