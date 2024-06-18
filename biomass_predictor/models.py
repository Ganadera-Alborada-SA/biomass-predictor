import numpy as np
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


def linear_model(X, y):
    y = y.values.ravel()
    reg = LinearRegression().fit(X, y)
    print(f"Trained linear model with R^2 {reg.score(X, y)} on training data")
    print(f"Crossvalidation  R^2 score {np.mean(cross_val_score(reg, X, y, cv=5))}")
    print(
        f"LOOCV RMSE score {-np.mean(cross_val_score(reg, X, y, cv=LeaveOneOut(), scoring='neg_root_mean_squared_error'))}"
    )

    print(f"Features: {list(X.columns)}")
    print(f"Coefficients: {reg.coef_}")
    print(f"Intercept: {reg.intercept_}")


def random_forest(X, y):
    y = y.values.ravel()
    reg = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
    print(f"Trained ensemble model with R^2 {reg.score(X, y)} on training data")
    print(f"Crossvalidation  R^2 score {np.mean(cross_val_score(reg, X, y, cv=5))}")
    print(
        f"LOOCV RMSE score {-np.mean(cross_val_score(reg, X, y, cv=LeaveOneOut() , scoring='neg_root_mean_squared_error'))}"
    )


def neural_network(X, y):
    y = y.values.ravel()
    pipe = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            (
                "NN",
                MLPRegressor(
                    hidden_layer_sizes=[5],
                    alpha=1,
                    learning_rate="constant",
                    learning_rate_init=0.6,
                    max_iter=40000,
                    momentum=0.3,
                    random_state=0,
                ),
            ),
        ]
    )

    reg = pipe.fit(X, y)
    print(f"Trained neural network model with R^2 {reg.score(X, y)} on training data")
    print(f"Crossvalidation  R^2 score {np.mean(cross_val_score(reg, X, y, cv=5))}")
    print(
        f"LOOCV RMSE score {-np.mean(cross_val_score(reg, X, y, cv=LeaveOneOut(), scoring='neg_root_mean_squared_error'))}"
    )
