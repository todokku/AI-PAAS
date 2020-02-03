"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

giving a rough estimate of fault of the signal

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class FaultDetection:

    def __init__(self, conf_fac):
        self.conf_fac = conf_fac
        self.co_eff = []
        self.ig_feature = []  # Ignored Features

    def _poly_fit(self, y):
        x = np.arange(len(y))

        return np.polyfit(x, y, 2)

    def get_faulty_cycles(self, input_df,
                          e_id_df):  # Provides an estimate for the number of faulty cycles in each engine

        no_engines = e_id_df.max()
        fault_st_mean = np.full(no_engines, 0)
        fault_st_std = np.full(no_engines, 0)

        input_df = pd.concat((e_id_df, input_df), axis=1)

        for i, in_e_df in input_df.groupby('Engine ID'):
            p_coeff = np.apply_along_axis(self._poly_fit, 0, in_e_df.iloc[:, 1:])

            b_array = np.all([-p_coeff[1, :] / (2 * p_coeff[0, :]) > 1,
                              -p_coeff[1, :] / (2 * p_coeff[0, :]) < in_e_df.shape[0]],
                             axis=0)
            p_coeff = p_coeff[:, b_array]

            self.co_eff.append(p_coeff)
            self.ig_feature.append(b_array)

            st_pts = np.apply_along_axis(np.roots, 0, p_coeff[:2, :])

            q25_75 = np.percentile(st_pts, [25, 75])

            iqr = q25_75[1] - q25_75[0]
            st_pts = st_pts[np.all([(st_pts < q25_75[1] + 1.5 * iqr),
                                    (st_pts > q25_75[0] - 1.5 * iqr)],
                                   axis=0)]

            fault_st_mean[i - 1] = st_pts.mean()
            fault_st_std[i - 1] = (st_pts.var()) ** 0.5

        self.fault_start = fault_st_mean + fault_st_std * self.conf_fac

        no_fault_cycles = np.round(input_df.groupby('Engine ID').size().values - self.fault_start).astype(int)

        return no_fault_cycles


if __name__ == '__main__':
    from GetCMAPSS import CMAPSS
    from DeNoising import DeNoiser
    from Normalising import Normalizer

    ds_no = 2

    raw_data = CMAPSS(ds_no)
    de_noise = DeNoiser(7, 3)
    raw_data.get_data()

    selected_feat = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31', 'W32']

    train_df = raw_data.Train_input[selected_feat]
    e_id = raw_data.Train_input['Engine ID']

    if ds_no == 2 or ds_no == 4:
        op_cond_df = raw_data.Train_input.iloc[:, 2:5]
        norm = Normalizer(6)
    else:
        op_cond_df = None
        norm = Normalizer()

    train_df = norm.normalising(train_df, op_cond_df)
    train_df = de_noise.smooth(train_df, e_id)

    fd = FaultDetection(-0.1)
    faulty_cycles = fd.get_faulty_cycles(train_df, e_id)

    engine_no = 1
    # Plotting all Features
    e_df = train_df.loc[e_id == engine_no, fd.ig_feature[engine_no - 1]]
    x = np.arange(e_df.shape[0])
    for i in range(0, e_df.shape[1]):
        plt.title(f'Engine Number {engine_no}')
        plt.plot(x, e_df.iloc[:, i])
        plt.plot(np.polyval(fd.co_eff[engine_no - 1][:, i], x))
        plt.ylabel(selected_feat[i - 1])
        plt.xlabel('Cycles')
        plt.show()
