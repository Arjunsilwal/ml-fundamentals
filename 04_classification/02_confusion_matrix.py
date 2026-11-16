# 04_classification/02_confusion_matrix.py
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

pipe = Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=1000))])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))