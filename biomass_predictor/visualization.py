import data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = data.get_biomass_growth_data()
    ax = df.plot.scatter(x="previous_biomass", y="biomass", c="DarkBlue")
    plt.show()
