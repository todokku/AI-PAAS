import scipy.signal as scipy_sig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DeNoiser:

    def __init__(self, window_length, poly_order):
        self.window_length = window_length
        self.poly_order = poly_order

    def _savgol(self, signal):
        smooth_sig = scipy_sig.savgol_filter(signal,
                                             self.window_length,
                                             self.poly_order,
                                             mode='nearest')
        return smooth_sig

    def smooth(self, input_df):
        e_id = input_df['Engine ID']

        input_df = input_df.groupby('Engine ID').transform(self._savgol)

        input_df = pd.concat((e_id, input_df), axis=1)
        return input_df


if __name__ == '__main__':
    from Input import CMAPSS

    raw_data = CMAPSS(1)
    de_noise = DeNoiser(7, 3)
    raw_data.get_data()

    selected_feat = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31', 'W32']

    raw_df = raw_data.Train_input[['Engine ID'] + selected_feat]
    smooth_df = de_noise.smooth(raw_df)

    engine_no = 6
    # Plotting all Features
    e_dfr = raw_df.loc[raw_df['Engine ID'] == engine_no, :]
    e_dfs = smooth_df.loc[raw_df['Engine ID'] == engine_no, :]
    for i in range(1, raw_df.shape[1]):
        plt.plot(e_dfr.iloc[:, i])
        plt.plot(e_dfs.iloc[:, i])
        plt.ylabel(selected_feat[i-1])
        plt.xlabel('Cycles')
        plt.show()
