from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def linear_model(X, y):
    reg = LinearRegression().fit(X, y)
    print(f"Trained linear model with R^2 {reg.score(X, y)} on training data")
    print(f"Features: {X.columns}")
    print(f"Coefficients: {reg.coef_}")
    print(f"Intercept: {reg.intercept_}")
    return reg


def random_forest(X, y):
    reg = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
    print(f"Trained ensemble model with R^2 {reg.score(X, y)} on training data")
