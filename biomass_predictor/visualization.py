import data
import matplotlib.pyplot as plt
import seaborn as sn


def visualize_train_data():
    df = data.get_growth_dataset()
    ax = df.plot.scatter(x="previous_biomass", y="biomass", c="DarkBlue")
    ax = df.plot.hist(column="biomass_change")
    plt.show()


def visualize_correlation_matrix():
    df = data.get_growth_dataset()
    df = df[
        [
            "previous_biomass",
            "cumulative_temperature",
            "cumulative_humidity",
            "cumulative_rain",
            "cumulative_light",
            "biomass",
        ]
    ]
    corr_matrix = df.corr()  # type: ignore
    ax = sn.heatmap(corr_matrix, annot=True)
    plt.tight_layout()

    plt.show()


def visualize_residuals(reg, X, y):
    y = y.values.ravel()
    y_pred = reg.predict(X)
    residuals = y - y_pred
    X["residual"] = residuals

    ax = X.plot.scatter(x="previous_biomass", y="residual", c="DarkBlue")
    ax = X.plot.scatter(x="cumulative_temperature", y="residual", c="DarkBlue")
    ax = X.plot.scatter(x="cumulative_humidity", y="residual", c="DarkBlue")
    ax = X.plot.scatter(x="cumulative_rain", y="residual", c="DarkBlue")
    ax = X.plot.scatter(x="cumulative_light", y="residual", c="DarkBlue")
    ax = X.plot.hist(column="residual")
    plt.show()


if __name__ == "__main__":
    visualize_train_data()
