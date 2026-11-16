from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipe = Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=1000))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
print(scores.round(3))                               
print(scores.mean().round(3), scores.std().round(3)) 