
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수, test_x는 테스트 데이터
# pandas의 DataFrame, Series의 자료형 사용(numpy의 array로 값을 저장하기도 함.)

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# 학습 데이터를 학습 데이터와 평가용 데이터셋으로 분할
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

import xgboost as xgb
from sklearn.metrics import log_loss

# 특징과 목적변수를 xgboost의 데이터 구조로 변환
# 학습 데이터의 특징과 목적변수는 tr_x, tr_y
# 검증 데이터의 특징과 목적변수는 va_x, va_y
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)

def fair(preds, dtrain):
    x = preds - dtrain.get_labels()
    c = 1.0
    den = abs(x) +c
    grad = c*x/den
    hess = c*c/den**2
    return grad, hess


def psuedo_huber(preds, dtrain):
    d = preds - dtrain.get_labels()
    delta = 1.0
    scale = 1+(d/delta)**2
    scale_sqrt = np.sqrt(scale)
    grad = d/scale_sqrt
    hess = 1/scale/scale_sqrt
    return grad, hess


