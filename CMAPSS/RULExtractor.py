"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""

import numpy as np
import matplotlib.pyplot as plt


def rul_generator(time_steps, const_rul, std_dev):
    rul_array = np.concatenate((np.repeat(const_rul, time_steps - const_rul), np.arange(const_rul - 1, -1, -1)))

    noise = np.random.randn(time_steps) * std_dev

    return rul_array + noise


def const_finder(input_array):
    initial = 10
    time_steps = input_array.size
    step_size = 5
    n_steps = np.floor((time_steps - initial) / step_size).astype(int)


if __name__ == '__main__':
    rul = rul_generator(232, 100, 30)
    #
    plt.plot(rul)
    plt.show()
    noise = np.random.randn(232) * 20
    initial = 15
    time_steps = rul.size
    step_size = 5
    n_steps = np.floor((time_steps - initial) / step_size).astype(int) + 1

    std_dev = np.array([])
    test = []
    coeff = []

    for i in range(n_steps):
        test.append(rul[:initial + step_size * i])
        coeff.append(np.polyfit(np.arange(rul[:initial + step_size * i].size), rul[:initial + step_size * i], 2))




