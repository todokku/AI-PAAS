# import seaborn as sns

from Input import Satellite

import matplotlib.pyplot as plt
import pandas as pd

Sat = Satellite('2009')

Sat.read_data()


def plot(series):
    plt.plot(series)
    plt.xlabel('TimeSteps')
    plt.ylabel(series.name)


plt.rcParams['figure.figsize'] = 20, 10

plot(Sat.input_data['   adcs_main ADRW2TCMD'])
plt.show()
plot(Sat.input_data.iloc[:, 1])
plt.show()
plot(Sat.input_data.iloc[:, 9])
plt.show()
plot(Sat.input_data.iloc[:, 13])
plt.show()
