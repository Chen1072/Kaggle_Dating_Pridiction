import numpy as np
import pandas as pd

w = np.array([[-0.05969438],
              [0.75124374],
              [0.38934646],
              [0.3183239],
              [0.51740783],
              [-0.19128925],
              [-0.58901905],
              [0.39864727],
              [-0.22065838],
              [0.11599619],
              [1.08380649],
              [-0.33331604],
              [0.43149528],
              [1.02157013],
              [1.31003441],
              [1.59949289]])
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

df = pd.read_csv("Thyroid_Diff.csv")
test_df = df[1:]
test_np = test_df.to_numpy()
t_test = test_np[:, -1].reshape(-1, 1)
X_test = test_np[:, :-1]
out = sigmoid(np.matmul(X_test, w))
out_binary = (out >= 0.5).astype(int)
combined_output = np.hstack((out_binary, t_test))

correct_predictions = np.sum(out_binary == t_test)
total_predictions = len(t_test)
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy:.2f} with {total_predictions} predictions")