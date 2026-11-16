# 01_ml_concepts/01_features_labels_split.py
from sklearn.model_selection import train_test_split

from make_churn import make_churn

df = make_churn()
print(df.head())                         # first 5 customers

X = df.drop(columns="churn")             # features: every column except the answer
y = df["churn"]                          # label: the answer column
print(X.shape, y.shape)                  # (2000, 5) (2000,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape)                        # (1600, 5) (400, 5)
print(y_train.mean().round(3), y_test.mean().round(3))    # 0.195 0.195