"""
AIAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AIAS Phd Student

"""
import numpy as np


class PrepCNNInOut:

    def __init__(self, window_len=30, stride=1):
        self.window_len = window_len
        self.stride = stride

    def _prep_train_inputs(self, input_array, e_id):
        train_seq = np.split(input_array, np.cumsum(np.unique(e_id, return_counts=True)[1])[:-1])
        no_features = input_array.shape[1]

        for i, seq in enumerate(train_seq):
            train_seq[i] = seq.reshape(1, -1, no_features)



        return train_seq
