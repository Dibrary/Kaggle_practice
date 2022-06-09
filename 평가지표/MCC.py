from sklearn.metrics import matthews_corrcoef

# tp = 70
# tn = 10
# fp = 10
# fn = 10

y_true = [1,0,1,1,0,1,1,0]
y_pred = [0,0,1,1,0,0,1,1]


print(matthews_corrcoef(y_true, y_pred))


