"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""
import numpy as np


class PrepRnnInOut:

    def __init__(self, faulty_start, s_len=2, val_split=0):
        self.s_len = s_len
        self.val_split = val_split
        self.faulty_start = faulty_start

    def _assign_dummy(self, x):
        x[x[0] + 1:] = 0.
        return x

    def _prep_train_inputs(self, input_array, e_id):
        cycle_len = np.unique(e_id, return_counts=True)[-1]
        self.no_ins = np.round((cycle_len - self.faulty_start) / self.s_len).astype(int)

        no_features = input_array.shape[1]
        engine_seq = np.split(input_array, np.cumsum(np.unique(e_id, return_counts=True)[1])[:-1])
        seq_list = []
        for i, seq in enumerate(engine_seq):
            engine_list = []
            for j in range(self.no_ins[i]):
                engine_list.append(np.split(seq, [cycle_len[i] - (j + 1) * self.s_len])[0].reshape(1,
                                                                                                        -1,
                                                                                                        no_features))
            seq_list.extend(engine_list)

        return seq_list

    def _prep_train_outputs(self, e_id):
        outputs = np.arange(self.s_len, self.s_len * (self.no_ins.max() + 1), self.s_len).reshape(-1, 1)
        outputs = np.repeat(outputs, e_id.max(), axis=1)
        outputs = np.concatenate((self.no_ins.reshape(1, -1), outputs), axis=0)
        outputs = np.apply_along_axis(self._assign_dummy, 0, outputs)
        outputs = outputs[1:, :]
        outputs = outputs.flatten('F')
        return outputs[outputs != 0.]  # Removing Padded Values

    def create_train_inputs(self, input_array, e_id):
        if self.val_split == 0:
            return self._prep_train_inputs(input_array, e_id), self._prep_train_outputs(e_id)

    def create_test_inputs(self, input_array, e_id):
        no_features = input_array.shape[1]
        seq_list = np.split(input_array, np.cumsum(np.unique(e_id, return_counts=True)[1])[:-1])
        for i, seq in enumerate(seq_list):
            seq_list[i] = seq.reshape(1, -1, no_features)
        return seq_list


if __name__ == '__main__':
    from GetCMAPSS import CMAPSS

    raw_data = CMAPSS(1)
    raw_data.get_data()

    selected_feat = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31', 'W32']
    train_df = raw_data.Train_input[selected_feat]
    test_df = raw_data.Test_input[selected_feat]
    e_id_df = raw_data.Train_input['Engine ID']
    e_id_test_df = raw_data.Test_input['Engine ID']
    preper = PrepRnnInOut(np.array([20] * e_id_df.max()))

    x, y = preper.create_train_inputs(train_df.to_numpy(), e_id_df.to_numpy())
    z = preper.create_test_inputs(test_df.to_numpy(), e_id_test_df.to_numpy())