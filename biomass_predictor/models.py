import numpy as np
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error

import optuna


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


class Objective:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        regressor_name = trial.suggest_categorical(
            "regressor", ["LinearRegression", "RandomForest", "MLP"]
        )
        if regressor_name == "LinearRegression":
            lr_alpha = trial.suggest_float("lr_alpha", 1e-10, 1000, log=True)
            regressor_obj = Pipeline(
                [("scaler", MinMaxScaler()), ("Linear", Lasso(alpha=lr_alpha))]
            )
        elif regressor_name == "RandomForest":
            rf_max_depth = trial.suggest_int("rf_max_depth", 1, 32, log=True)
            rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 100, log=True)
            rf_max_features = trial.suggest_int("rf_max_features", 1, 4, log=True)
            regressor_obj = RandomForestRegressor(
                max_depth=rf_max_depth,
                n_estimators=rf_n_estimators,
                max_features=rf_max_features,
                random_state=0,
            )
        else:
            mlp_number_hidden_layers = trial.suggest_int(
                "mlp_number_hidden_layers", 1, 3
            )
            mlp_alpha = trial.suggest_float("mlp_alpha", 1e-5, 1000, log=True)
            mlp_learning_rate = trial.suggest_categorical(
                "mlp_learning_rate", ["constant", "invscaling", "adaptive"]
            )
            mlp_learning_rate_init = trial.suggest_float(
                "mlp_learning_rate_init", 1e-2, 0.7, log=True
            )
            mlp_momentum = trial.suggest_float("mlp_momentum", 0, 0.9, step=0.1)
            regressor_obj = Pipeline(
                [
                    ("scaler", MinMaxScaler()),
                    (
                        "NN",
                        MLPRegressor(
                            hidden_layer_sizes=[5] * mlp_number_hidden_layers,
                            alpha=mlp_alpha,
                            learning_rate=mlp_learning_rate,
                            learning_rate_init=mlp_learning_rate_init,
                            max_iter=500000,
                            momentum=mlp_momentum,
                            random_state=0,
                        ),
                    ),
                ]
            )

        score = np.mean(cross_val_score(regressor_obj, self.X, self.y, cv=5))
        loocv_score = -np.mean(
            cross_val_score(
                regressor_obj,
                self.X,
                self.y,
                cv=LeaveOneOut(),
                scoring="neg_root_mean_squared_error",
            )
        )
        trial.set_user_attr("loocv_score", loocv_score)
        return score


def get_best_model(X, y):
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="biomass_prediction",
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        study.optimize(Objective(X, y.values.ravel()), n_trials=50)

    reg = None
    print(study.best_params)
    match study.best_params["regressor"]:
        case "LinearRegression":
            reg = Lasso(alpha=study.best_params["lr_alpha"])
        case "RandomForest":
            reg = RandomForestRegressor(
                max_depth=study.best_params["rf_max_depth"],
                n_estimators=study.best_params["rf_n_estimators"],
                max_features=study.best_params["rf_max_features"],
            )
        case "MLP":
            reg = MLPRegressor(
                hidden_layer_sizes=[5] * study.best_params["mlp_number_hidden_layers"],
                alpha=study.best_params["mlp_alpha"],
                learning_rate=study.best_params["mlp_learning_rate"],
                learning_rate_init=study.best_params["mlp_learning_rate_init"],
                max_iter=50000,
                momentum=study.best_params["mlp_momentum"],
                random_state=0,
            )

    reg.fit(X, y.values.ravel())  # type: ignore
    return reg


def evaluate_model(reg, X_test, y_test):
    y_test = y_test.values.ravel()
    print(f"Test  R^2 score {reg.score(X_test, y_test)}")
    y_pred = reg.predict(X_test)
    print(f"Test RMSE score {root_mean_squared_error(y_test, y_pred)}")
