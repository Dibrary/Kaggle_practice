
from sklearn.metrics import f1_score
import numpy as np

y_true = np.array([[1,1,0],
                   [1,0,0],
                   [1,1,1],
                   [0,1,1],
                   [0,0,1]])

y_pred = np.array([[1,0,1],
                   [0,1,0],
                   [1,0,1],
                   [0,0,1],
                   [0,0,1]])

mean_f1 = np.mean([f1_score(y_true[i,:], y_pred[i,:]) for i in range(len(y_true))])


n_class = 3
macro_f1 = np.mean([f1_score(y_true[:, c], y_pred[:, c]) for c in range(n_class)])


micro_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))

print(mean_f1, macro_f1, micro_f1)




# 이렇게도 계산할 수 있다.

mean_f1 = f1_score(y_true, y_pred, average='samples')
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

print(mean_f1, macro_f1, micro_f1)




