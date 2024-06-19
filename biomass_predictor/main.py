import data
import matplotlib.pyplot as plt
import models

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data = data.get_growth_dataset()
    print(len(data))
    # print(data.head())
    X = data[
        [
            "previous_biomass",
            "cumulative_temperature",
            "cumulative_humidity",
            "cumulative_rain",
            "cumulative_light",
        ]
    ]
    y = data[["biomass"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # models.linear_model(X_train, y_train)
    # models.random_forest(X_train, y_train)
    # models.neural_network(X_train, y_train)

    reg = models.get_best_model(X_train, y_train)
    models.evaluate_model(reg, X_test, y_test)
