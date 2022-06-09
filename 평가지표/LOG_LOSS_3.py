
# 3차원

from sklearn.metrics import log_loss
import numpy as np

y_true = np.array([0,2,1,2,2])
y_pred = np.array([[0.68, 0.32, 0.00],
                   [0.00,0.00,1.00],
                   [0.6, 0.4, 0.0],
                   [0.0, 0.0, 1.0],
                   [0.28, 0.12, 0.6]])

logloss = log_loss(y_true, y_pred)
print(logloss)

