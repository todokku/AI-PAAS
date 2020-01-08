# -*- coding: utf-8 -*-
"""
AIAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AIAS Phd Student

"""

# =================================================================================================
# Preprocessing Module!
# TODO Vectorize the code!
# =================================================================================================

# Libraries

import scipy.signal          as scipy_sig
import numpy                 as np
import sklearn.decomposition as skl_d
import sklearn.cluster       as skl_c


class cMAPSS:

    def __init__(self,
                 win_len=21,
                 p_order=3,
                 std_fac=0,  # Std factor. Recommended to choose value from -1 to 0
                 s_len=2,  # Length of Stagger // Unit - Cycle
                 pca_var=0.97,
                 no_splits=5,  # k splits cross validation
                 threshold=1e-5,
                 denoising=True,
                 even_split=True,
                 multi_op_normal=True,  # Used to enable multi operating condition normalising and remove opcond
                 preptype='tj'):

        self.win_len = win_len
        self.p_order = p_order
        self.std_fac = std_fac
        self.s_len = s_len
        self.pca_var = pca_var
        self.no_splits = no_splits
        self.threshold = threshold
        self.denoising = denoising
        self.even_split = even_split
        self.multi_op_normal = multi_op_normal
        self.preptype = preptype

        self.no_opcond = 6  # Temporary assignment , change to if statement or lookup table

    # ================================================================================================

    def preprocess(self,
                   input_data,
                   isTrain=True,
                   force_feat=0):

        self._isTrain = isTrain
        self._force_feat = force_feat
        self._input_data = input_data

        self.no_engines = self._input_data.iloc[-1, 0]
        self._max_cycles = self._input_data['Cycles'].max()
        self._e_id = self._input_data.iloc[:, 0]
        self._cycles = self._input_data.iloc[:, 1]
        self._opcond = self._input_data.iloc[:, 2:5]
        self._input_data = self._input_data.iloc[:, 5:]
        self._cycle_len = self._e_id.value_counts().sort_index().to_numpy()

        if self._isTrain:
            self.data_var = self._input_data.var()
            self._input_data = self._input_data.loc[:, self.data_var > self.threshold]

            self.opcond_var = self._opcond.var()
            self._opcond = self._opcond.loc[:, self.opcond_var > self.threshold]

        else:
            self._input_data = self._input_data.loc[:, self.data_var > self.threshold]
            self._opcond = self._opcond.loc[:, self.opcond_var > self.threshold]

        self.normalising()

        if self.denoising:
            for i in range(1, self.no_engines + 1):
                self._input_data.loc[self._e_id == i, :] = self._input_data.loc[self._e_id == i, :].apply(self._savgol)

        if self._isTrain:
            self.get_fcycles()

        self._input_data = self._input_data.to_numpy()

        self.dim_red()
        self.RNN_prep()

    # ================================================================================================

    def RNN_prep_constRUL(self):
        pass

    # ================================================================================================

    def _savgol(self, signal):

        smooth_sig = scipy_sig.savgol_filter(signal,
                                             self.win_len,
                                             self.p_order,
                                             mode='nearest')
        return smooth_sig

    # ================================================================================================

    def dim_red(self):

        if self._isTrain:

            pca = skl_d.PCA(n_components=self.pca_var,
                            svd_solver='full')
            self._input_data = pca.fit_transform(self._input_data)
            self.features = pca.n_components_

            print(f'\nNumber of extracted features are {self.features}')

        else:
            try:
                features = self.features
            except AttributeError:
                if self._force_feat == 0:
                    raise Exception("Please run train_data first")
                else:
                    features = self._force_feat

            pca = skl_d.PCA(n_components=features)
            self._input_data = pca.fit_transform(self._input_data)

            self.test_pca = round(pca.explained_variance_ratio_.sum(), 2)
            if self.test_pca < self.pca_var:
                print(f'PCA test variation is less than the train variation. It is - {self.test_pca}')

    # ================================================================================================

    def RNN_prep(self):  # Preparing the data for any RNN

        if self._isTrain:

            self._no_ins = np.round(self.no_fcycles / self.s_len)  # fcycles are faulty cycles
            self._no_ins = self._no_ins.astype(int) - 1  # the rul 0 is removed
            self._no_ins = self._no_ins.reshape(-1, 1)

            # Generating train_out through vectorising
            outputs = np.arange(self.s_len, self.s_len * (self._no_ins.max() + 1), self.s_len).reshape(-1, 1)

            outputs = np.repeat(outputs, self.no_engines, axis=1)
            outputs = np.concatenate((self._no_ins.T, outputs), axis=0)
            outputs = np.apply_along_axis(self._assign_dummy, 0, outputs)
            outputs = outputs[1:, :]
            outputs = outputs.flatten('F')
            temp = outputs[outputs != 1000]  # Removing Padded Values
            outputs = temp

            if self.even_split:
                self.train_prepES(outputs)
            else:
                self.train_prep(outputs)

        else:
            self.test_in = np.full((self.no_engines,
                                    self._max_cycles,
                                    self._input_data.shape[1]),
                                   1000.0)

            for i in range(self.no_engines):
                c_len = self._cycle_len[i]
                temp = self._input_data[self._e_id == i + 1, :]
                self.test_in[i, :c_len, :] = temp

    # ================================================================================================

    def train_prepES(self, outputs):

        split_no_ins = np.floor(self._no_ins / self.no_splits).astype(int).reshape(-1)
        split_last_ins = split_no_ins.cumsum()
        split_first_ins = np.append(0, split_no_ins[:-1]).cumsum()

        fsplit_no_ins = self._no_ins.reshape(-1) - split_no_ins * (self.no_splits - 1)
        fsplit_last_ins = fsplit_no_ins.cumsum()
        fsplit_first_ins = np.append(0, fsplit_no_ins[:-1]).cumsum()
        first_ins = np.append(0, self._no_ins[:-1]).cumsum()

        total_no_ins = self._no_ins.sum()
        split_tno_ins = split_no_ins.sum()
        fsplit_tno_ins = total_no_ins - split_tno_ins * (self.no_splits - 1)

        self.splits_in = []
        self.splits_out = []

        for k in range(self.no_splits - 1):
            self.splits_in.append(np.full((split_tno_ins,
                                           self._max_cycles,
                                           self._input_data.shape[1]),
                                          1000.0))

            self.splits_out.append(np.full(split_tno_ins, 1000.0))

        self.splits_in.append(np.full((fsplit_tno_ins,
                                       self._max_cycles,
                                       self._input_data.shape[1]),
                                      1000.0))
        self.splits_out.append(np.full(fsplit_tno_ins, 1000.0))

        for nins, first, sins, sfins, sfirst, slast, sffirst, sflast, eng_cycle, i in zip(self._no_ins,
                                                                                          first_ins,
                                                                                          split_no_ins,
                                                                                          fsplit_no_ins,
                                                                                          split_first_ins,
                                                                                          split_last_ins,
                                                                                          fsplit_first_ins,
                                                                                          fsplit_last_ins,
                                                                                          self._cycle_len,
                                                                                          range(1,
                                                                                                self.no_engines + 1)):

            indexes = np.arange(nins)
            np.random.shuffle(indexes)

            temp = self._input_data[self._e_id == i, :]

            for k in range(self.no_splits - 1):

                for ind, j in zip(indexes[sins * k: sins * (k + 1)], range(sfirst, slast)):
                    self.splits_in[k][j, :eng_cycle - (ind + 1) * self.s_len, :] = temp[:-(ind + 1) * self.s_len, :]

                self.splits_out[k][np.arange(sfirst, slast)] = outputs[indexes[sins * k: sins * (k + 1)]]

            for ind, j in zip(indexes[sins * (self.no_splits - 1):], range(sffirst, sflast)):
                self.splits_in[self.no_splits - 1][j, :eng_cycle - (ind + 1) * self.s_len, :] = temp[:-(
                        ind + 1) * self.s_len, :]

            self.splits_out[self.no_splits - 1][np.arange(sffirst, sflast)] = outputs[
                indexes[sins * (self.no_splits - 1):]]

    # ================================================================================================

    # def train_prep(self, outputs):
    #
    #     total_no_ins = self._no_ins.sum()
    #     split_no_ins = np.round(total_no_ins/self.no_splits).astype(int)
    #
    #     for eng_cycle, i in zip(self._cycle_len, range(1, self.no_engines + 1)):
    #
    #         for k in range(self.no_splits - 1):
    #         temp = self._input_data[self._e_id == i, :]

    # ================================================================================================

    def get_fcycles(self):  # Provides an estimate for the number of faulty cycles in each engine

        def _poly_fit(y):
            x = np.arange(len(y))

            return np.polyfit(x, y, 2)

        fstart_mean = np.full(self.no_engines, 0)
        fstart_std = np.full(self.no_engines, 0)

        for i in range(self.no_engines):
            p_coeff = np.apply_along_axis(_poly_fit, 0, self._input_data.loc[self._e_id == i + 1, :])

            temp = np.all([-p_coeff[1, :] / (2 * p_coeff[0, :]) > 1,
                           -p_coeff[1, :] / (2 * p_coeff[0, :]) < self._cycle_len[i]],
                          axis=0)

            temp = p_coeff[:, temp]
            p_coeff = temp
            st_pts = np.apply_along_axis(np.roots, 0, p_coeff[:2, :])

            q25_75 = np.percentile(st_pts, [25, 75])

            iqr = q25_75[1] - q25_75[0]
            temp = st_pts[np.all([(st_pts < q25_75[1] + 1.5 * iqr),
                                  (st_pts > q25_75[0] - 1.5 * iqr)],
                                 axis=0)]
            st_pts = temp

            fstart_mean[i] = st_pts.mean()
            fstart_std[i] = (st_pts.var()) ** 0.5

        self.fstart = fstart_mean + fstart_std * self.std_fac

        self.no_fcycles = self._cycle_len - self.fstart
        self.no_fcycles = np.round(self.no_fcycles).astype(int)

    # ================================================================================================

    def normalising(self):

        if self._opcond.shape[1] != 0:

            if self._isTrain:
                self.cluster = skl_c.KMeans(self.no_opcond, random_state=0).fit(self._opcond)

            op_state = self.cluster.predict(self._opcond)

            for i in range(self.no_opcond):
                self._input_data.loc[op_state == i, :] = self._input_data.loc[op_state == i, :].apply(
                    lambda x: (x - x.mean()) / x.std())

            self._input_data = self._input_data.dropna(axis='columns')

        else:
            self._input_data = self._input_data.apply(lambda x: (x - x.mean()) / x.std())

    # ================================================================================================

    def _assign_dummy(self, x):

        x[x[0] + 1:] = 1000

        return x

    # ================================================================================================

    def report_card(self):
        pass


# ================================================================================================
# ================================================================================================            

if __name__ == '__main__':
    from Input import cMAPSS as ci

    ci.get_data(2)
    pp1 = cMAPSS()
    pp1.preprocess(ci.Train_input)
#    pp1.preprocess(ci.Test_input, False)
