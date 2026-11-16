# 05_decision_trees/02_pruning.py
from common import *
from sklearn.tree import DecisionTreeClassifier

tree_prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

A = tree_prep.fit_transform(X_train)     # learn the cleaning from train, apply to train
B = tree_prep.transform(X_test)          # apply the same cleaning to test

for alpha in [0, 0.001, 0.002, 0.005]:
    t = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(A, y_train)
    print(alpha, t.get_n_leaves(), round(t.score(A, y_train), 3), round(t.score(B, y_test), 3))