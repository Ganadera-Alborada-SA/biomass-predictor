import data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = data._get_biomass_growth_data()
    ax = df.plot.scatter(x="previous_biomass", y="biomass", c="DarkBlue")
    ax = df.plot.hist(column="biomass_change")
    plt.show()
