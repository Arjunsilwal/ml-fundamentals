# 05_decision_trees/03_feature_importance.py
import pandas as pd

from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

tree_prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

rf = Pipeline([
    ("prep", tree_prep),
    ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42)),
]).fit(X_train, y_train)

# Method 1: impurity based, built into the model
names = rf.named_steps["prep"].get_feature_names_out()
print(pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False))

# Method 2: permutation importance, measured on the test set
perm = permutation_importance(rf, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42)
print(pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False))