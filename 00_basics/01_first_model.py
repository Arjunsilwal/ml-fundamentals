import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.DataFrame({
    "tenure":          [2, 5, 3, 40, 60, 48, 4, 55],
    "monthly_charges": [90, 85, 95, 40, 35, 50, 80, 45],
    "churn":           [1, 1, 1, 0, 0, 0, 1, 0],
})

X = data[["tenure", "monthly_charges"]] 
y = data["churn"]


model = DecisionTreeClassifier(random_state=42)

model.fit(X, y)

new_customers = pd.DataFrame({"tenure": [3, 50], "monthly_charges": [88, 42]})

print(model.predict(new_customers))
