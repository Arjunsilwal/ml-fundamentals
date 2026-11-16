from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

X_d, y_d = load_diabetes(return_X_y=True)
Xa, Xb, ya, yb = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

reg = LinearRegression().fit(Xa, ya)
pred = reg.predict(Xb)
print(round(mean_absolute_error(yb, pred), 1))        
print(round(root_mean_squared_error(yb, pred), 1))    
print(round(r2_score(yb, pred), 3))                