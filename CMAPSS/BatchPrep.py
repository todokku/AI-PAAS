"""
AIAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AIAS Phd Student

"""
import numpy as np


class PrepFixedInOut:

    def __init__(self, window_len=30, stride=1, flatten=False):
        self.window_len = window_len
        self.stride = stride
        self.flatten = flatten

    def _prep_train_inputs(self, input_array, e_id):
        train_seq = np.split(input_array, np.cumsum(np.unique(e_id, return_counts=True)[1])[:-1])
        no_features = input_array.shape[1]
        train_batch = np.zeros((1, self.window_len, no_features))

        for train_array in train_seq:

            n_strides_f = (train_array.shape[0]) // self.stride - self.window_len // self.stride + 1

            for i in range(n_strides_f):
                train_batch = np.append(train_batch, train_array[i * self.stride:i * self.stride + self.window_len],
                                        axis=0)

            n_strides_r = 0  # some value

            for i in range(n_strides_r):
                train_batch = np.append(train_batch,
                                        train_array[-(i * self.stride + self.window_len):-(i * self.stride)],
                                        axis=0)
        if self.flatten:
            pass  # input flatten code
        else:
            return train_batch[1:, :, :]
