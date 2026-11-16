from common import *
from sklearn.linear_model import LogisticRegression

model = Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=1000))])
model.fit(X_train, y_train)

print(model.predict(X_test[:3]))         
print(model.predict_proba(X_test[:3]))  