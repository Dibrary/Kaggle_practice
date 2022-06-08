
from sklearn.metrics import mean_squared_error
import numpy as np

y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 값이 0.55317 나온다.
# 그런지 직접 확인해보자.

total = 0
for i in range(len(y_true)):
    total += (y_true[i]-y_pred[i])**2
print(np.sqrt(total/len(y_true)))

# 같게 나온다.

