
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_x[num_cols])

train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

