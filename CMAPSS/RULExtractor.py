"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""

import numpy as np
import matplotlib.pyplot as plt


class RULExtractor:

    def __init__(self, initial=10, step_size=5):
        self.initial = initial
        self.step_size = step_size

    def extract_ruls(self, input_list):

        rul = np.array([])

        for input_array in input_list:

            n_steps = (input_array.size - self.initial) // self.step_size
            error = np.array([])
            h_rul = np.array([])

            for i in range(n_steps):
                const_array = input_array[:self.initial + i * self.step_size]
                linear_array = input_array[self.initial + i * self.step_size:]

                linear_array[0] = const_array.mean()
                weights = np.repeat(1, linear_array.size)
                weights[0] = 100

                x = np.arange(1, linear_array.size + 1)
                coeff = np.polynomial.polynomial.Polynomial.fit(x,
                                                                linear_array,
                                                                1,
                                                                w=weights).coef

                error = np.append(error, const_array.var() + (linear_array - coeff[0]*x - np.repeat(coeff[1], x.size).var()))
                h_rul = np.append(h_rul, coeff[0]*x[-1]+coeff[1])

            rul = np.append(rul, h_rul[np.argmin(error)])

        return rul


if __name__ == '__main__':

    def rul_generator(t_s, const_rul, std_dev):
        rul_array = np.concatenate((np.repeat(const_rul, t_s - const_rul), np.arange(const_rul - 1, -1, -1)))

        ns = np.random.randn(t_s) * std_dev

        return rul_array + ns


    test_list = []

    time_steps = [100, 200, 250, 175, 300]
    c_rul = [50, 100, 100, 75, 200]

    for i in range(5):
        test_list.append(rul_generator(time_steps[i], c_rul[i], 20))

    extractor = RULExtractor(10, 5)

    answer = extractor.extract_ruls(test_list)
