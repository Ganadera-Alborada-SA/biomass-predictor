import data
import matplotlib.pyplot as plt
import models

if __name__ == "__main__":
    data = data.get_growth_dataset()
    print(data)
    X = data[["biomass"]]
    y = data[
        [
            "previous_biomass",
            "mean_temperature",
            "mean_humedad",
            "total_rain",
            "total_light",
        ]
    ]
    models.linear_model(X, y)
    models.random_forest(X, y)
