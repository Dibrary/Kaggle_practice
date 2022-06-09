
from sklearn.metrics import log_loss
import numpy as np

y_true = [1,0,1,1,0,1]
y_pred = [0.1,0.2,0.8,0.8,0.1,0.3]

logloss = log_loss(y_true, y_pred)
print(logloss)


total = 0
for i in range(len(y_true)):
    total += (y_true[i]*np.log(y_pred[i]) + (1-y_true[i])*np.log(1-y_pred[i]))
print((total/len(y_true))*-1)


