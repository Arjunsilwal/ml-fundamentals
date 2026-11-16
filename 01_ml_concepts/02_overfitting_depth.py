from common import *
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

for depth in [1, 2, 3, 5, 8, 12, None]:
    model = Pipeline([
        ("prep", make_prep()),
        ("model", DecisionTreeClassifier(max_depth=depth, random_state=42)),
    ]).fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(depth, round(train_auc, 3), round(test_auc, 3))