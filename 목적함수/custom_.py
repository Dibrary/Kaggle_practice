
import xgboost as xgb
from sklearn.metrics import log_loss
import numpy as np

# tr_x = np.array([1,0,1,0,0,1,0])
# tr_y = np.array([0,0,0,1,0])
#
# va_x = np.array([1,1,0,1,0,0,1])
# va_y = np.array([1,1,0,1,0])
# 2차원 데이터가 주어져야 한다 ;; ValueError: Please reshape the input data into 2-dimensional matrix.

dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0/ (1.0 + np.exp(-preds))

    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'custom-error', float(sum(labels != (preds > 0.0)))/ len(labels)

params = {'verbosity':0, 'random_state':71}
num_round = 50
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

bst = xgb.train(params, dtrain, num_round, watchlist, obj=logregobj, feval = evalerror)

pred_val = bst.predict(dvalid)
pred = 1.0/(1.0 + np.exp(-pred_val))

logloss = log_loss(va_y, pred)
print(logloss)

params = {'verbosity':0, 'random_state':71, 'objective': 'binary:logistic'}
bst = xgb.train(params, dtrain, num_round, watchlist)

pred = bst.predict(dvalid)
logloss = log_loss(va_y, pred)
print(logloss)








