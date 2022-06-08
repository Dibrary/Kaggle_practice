from sklearn.metrics import confusion_matrix
import numpy as np

y_true = [1,0,1,1,0,1,1,0]
y_pred = [0,0,1,1,0,0,1,1]

tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp,fp], [fn, tn]])
print(confusion_matrix1)

confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)

