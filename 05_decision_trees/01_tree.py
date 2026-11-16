from common import *
from sklearn.tree import DecisionTreeClassifier, export_text

tree_prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),          # no scaling: trees do not need it
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

tree = Pipeline([
    ("prep", tree_prep),
    ("model", DecisionTreeClassifier(max_depth=2, random_state=42)),
]).fit(X_train, y_train)

names = list(tree.named_steps["prep"].get_feature_names_out())
print(export_text(tree.named_steps["model"], feature_names=names, show_weights=True))