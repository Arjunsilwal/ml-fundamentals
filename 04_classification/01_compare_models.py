from common import *
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

models = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42),
    "gb": GradientBoostingClassifier(random_state=42),
}

for name, clf in models.items():
    pipe = Pipeline([("prep", make_prep()), ("model", clf)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    print(name,
          round(accuracy_score(y_test, pred), 3),
          round(precision_score(y_test, pred), 3),
          round(recall_score(y_test, pred), 3),
          round(f1_score(y_test, pred), 3),
          round(roc_auc_score(y_test, proba), 3))