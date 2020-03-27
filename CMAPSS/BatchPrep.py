"""
AIAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AIAS Phd Student

"""
import numpy as np


class PrepFixedInOut:

    def __init__(self, window_len=30, flatten=False):
        self.window_len = window_len
        self.flatten = flatten

    def prep_train_inputs(self, input_array, e_id):
        input_list = np.split(input_array, np.cumsum(np.unique(e_id, return_counts=True)[1])[:-1])
        train_batch = np.empty((1, self.window_len, input_list[0].shape[1]))

        for input_array in input_list:
            batches = input_array.shape[0] - self.window_len + 1
            temp_array = np.repeat(input_array[np.newaxis, :, :], batches, axis=0)
            for index in range(batches):
                temp_array[index, :, :] = np.roll(temp_array[index, :, :], -1 * index, axis=0)

            train_batch = np.append(train_batch, temp_array[:, :self.window_len, :], axis=0)

        if self.flatten:
            pass  # input flatten code
        else:
            return train_batch[1:, :, :]

    def prep_train_outputs(self, fault_start, e_id):
        self.max_rul = np.unique(e_id, return_counts=True)[-1] - fault_start
        output_seq = []

        for i in range(e_id.max()):
            output_seq.append(np.concatenate((np.repeat(self.max_rul[i], fault_start[i] - self.window_len + 1),
                                              np.arange(self.max_rul[i] - 1, -1, -1)), axis=0))
        return np.concatenate(tuple(output_seq))
