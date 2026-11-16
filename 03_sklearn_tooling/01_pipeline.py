from common import *
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("prep", make_prep()),
    ("model", LogisticRegression(max_iter=1000)),
])

pipe.fit(X_train, y_train)               # fits imputer, scaler, encoder and model on train only
print(pipe.score(X_test, y_test))        # transforms test with what was learned, then predicts
print(pipe.named_steps["prep"].get_feature_names_out())