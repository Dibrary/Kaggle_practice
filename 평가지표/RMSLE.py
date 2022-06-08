
from sklearn.metrics import mean_squared_log_error
import numpy as np

y_true = [100, 0, 400]
y_pred = [200, 10, 200]

print(np.sqrt(mean_squared_log_error(y_true, y_pred)))

# 결과는 1.494 나온다.

total = 0
for i in range(len(y_true)):
    total += ((np.log(1+y_true[i]))-(np.log(1+y_pred[i])))**2

print(np.sqrt(total/len(y_true)))

# 직접 구해도 같은 값이 나온다.
