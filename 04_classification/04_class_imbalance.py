# 04_classification/04_class_imbalance.py
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

for weight in [None, "balanced"]:
    pipe = Pipeline([
        ("prep", make_prep()),
        ("model", LogisticRegression(max_iter=1000, class_weight=weight)),
    ]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(weight, "caught:", tp, "false alarms:", fp,
          "precision:", round(precision_score(y_test, pred), 3),
          "recall:", round(recall_score(y_test, pred), 3),
          "auc:", round(roc_auc_score(y_test, proba), 3))