"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""

import sklearn.cluster as skl_c


class Normalizer:

    def __init__(self, no_op_cond=None, cluster=None):

        self.no_op_cond = no_op_cond
        self.cluster = cluster

    def normalise(self, input_df, op_cond_df = None):

        if self.no_op_cond == 1:

            output_df = input_df.apply(lambda x: (x - x.mean()) / x.std())

        else:

            if op_cond_df is None:
                raise Exception('Must provide op conditions')
            if self.cluster is None:
                self.cluster = skl_c.KMeans(self.no_op_cond, random_state=0).fit(op_cond_df)

            op_state = self.cluster.predict(op_cond_df)
            output_df = input_df.copy()
            for i in range(self.no_op_cond):
                output_df.loc[op_state == i, :] = input_df.loc[op_state == i, :].apply(
                    lambda x: (x - x.mean()) / x.std())

            output_df = output_df.apply(lambda x: (x - x.mean()) / x.std())

        return output_df


if __name__ == '__main__':
    from GetCMAPSS import CMAPSS
    import matplotlib.pyplot as plt

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
    e_id_df = data.Train_input['Engine ID']
    train_df = norm.normalising(train_df, op_cond_df)

    engine_no = 5
    feature_no = 5
    plt.plot(data.Train_input.loc[e_id_df == engine_no, selected_feat[feature_no]])
    plt.show()
    plt.plot(train_df.loc[e_id_df == engine_no, selected_feat[feature_no]])
    plt.show()
