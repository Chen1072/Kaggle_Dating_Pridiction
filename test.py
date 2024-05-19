import numpy as np
import pandas as pd

w = [[-0.05969438]
     [0.75124374]
     [0.38934646]
     [0.3183239]
     [0.51740783]
     [-0.19128925]
     [-0.58901905]
     [0.39864727]
     [-0.22065838]
     [0.11599619]
     [1.08380649]
     [-0.33331604]
     [0.43149528]
     [1.02157013]
     [1.31003441]
     [1.59949289]]

df = pd.read_csv("Thyroid_Diff.csv")
split_point = int(len(df) * 0.99)
test_df = df[split_point:]
test_np = test_df.to_numpy()
t_test = test_np[:, -1].reshape(-1, 1)
X_test = test_np[:, :-1]
print(X_test @ w)
print(t_test)