"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""

import numpy as np
import matplotlib.pyplot as plt


class RULExtractor:

    def __init__(self):
        pass

    def extract_ruls(self, input_list):
        rul_array = np.array([])

        for input_array in input_list:
            p_coeff = np.polyfit(np.arange(1, input_array.size + 1), input_array, 2)
            rul_array = np.append(rul_array, np.polyval(p_coeff, input_array.size))

        return rul_array


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

    extractor = RULExtractor()

    answer = extractor.extract_ruls(test_list)
