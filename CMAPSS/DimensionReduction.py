"""
AI-AS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-AS Phd Student

"""

import sklearn.decomposition as skl_d
import matplotlib.pyplot as plt


class DimReduction:

    def __init__(self, var_threshold, no_features=None):

        self.var_threshold = var_threshold
        self.no_features = no_features

    def reduce_dimensions(self, input_array):

        if self.no_features is None:

            pca = skl_d.PCA(n_components=self.var_threshold, svd_solver='full')
            input_array = pca.fit_transform(input_array)
            self.no_features = pca.n_components_

            print(f'\nNumber of extracted features are {self.no_features}')

        else:

            pca = skl_d.PCA(n_components=self.no_features)
            input_array = pca.fit_transform(input_array)

            self.test_var = round(pca.explained_variance_ratio_.sum(), 2)
            if self.test_var < self.var_threshold:
                print(f'PCA test variation is less than the train variation. It is - {self.var_threshold}')

        return input_array


if __name__ == '__main__':
    from Input import CMAPSS
    from Normalising import Normalizer
    from DeNoising import DeNoiser

    ds_no = 2

    data = CMAPSS(2)
    data.get_data()

    selected_feat = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31', 'W32']

    if ds_no == 2 or ds_no == 4:
        op_cond_df = data.Train_input.iloc[:, 2:5]
        norm = Normalizer(6)
    else:
        op_cond_df = None
        norm = Normalizer()

    train_df = data.Train_input[selected_feat]
    e_id = data.Train_input['Engine ID']

    train_df = norm.normalising(train_df, op_cond_df)

    de_noise = DeNoiser(7, 3)
    train_df = de_noise.smooth(train_df, e_id)

    dreduce = DimReduction(0.7)
    train_array = dreduce.reduce_dimensions(train_df.to_numpy())

    engine_no = 6
    # Plotting all Features
    e_df = train_array[data.Train_input['Engine ID'] == engine_no, :]
    for i in range(0, train_array.shape[1]):
        plt.title(f'Engine Number {engine_no}')
        plt.plot(e_df[:, i])
        plt.ylabel(f'PC{i+1}')
        plt.xlabel('Cycles')
        plt.show()
