# 04_classification/03_threshold.py
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

pipe = Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=1000))])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]

for threshold in [0.5, 0.3, 0.2]:
    pred = (proba >= threshold).astype(int) 
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(threshold, "caught:", tp, "false alarms:", fp,
          "precision:", round(precision_score(y_test, pred), 3),
          "recall:", round(recall_score(y_test, pred), 3))