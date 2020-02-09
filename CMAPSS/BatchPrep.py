"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""
import numpy as np


class PrepRnnInOut_no_seq:

    def __init__(self, s_len=5, initial_cutoff=0., ins_dropped=0.):
        self.s_len = s_len
        self.initial_cutoff = initial_cutoff
        self.ins_dropped = ins_dropped
        self.no_ins = None
        self.cycle_len = None

    def _assign_dummy(self, x):
        x[x[0] + 1:] = 0.
        return x

    def _prep_train_inputs(self, fault_start, input_array, e_id):
        self.cycle_len = np.unique(e_id, return_counts=True)[-1]
        self.no_ins = np.round((self.cycle_len - fault_start) * (1-self.ins_dropped) / self.s_len).astype(int)
        no_features = input_array.shape[1]
        engine_seq = np.split(input_array, np.cumsum(np.unique(e_id, return_counts=True)[1])[:-1])
        seq_list = []
        for i, seq in enumerate(engine_seq):
            engine_list = []
            for j in range(self.no_ins[i]):
                engine_list.append(np.split(seq, [int(round(fault_start[i] * self.initial_cutoff)),
                                                  self.cycle_len[i] - (j + 1) * self.s_len])[1].reshape(1,
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

    def _create_train_inputs(self, fault_start, input_array, e_id):
        return self._prep_train_inputs(fault_start, input_array, e_id), self._prep_train_outputs(e_id)

    def _create_test_inputs(self, input_array, e_id):
        no_features = input_array.shape[1]
        seq_list = np.split(input_array, np.cumsum(np.unique(e_id, return_counts=True)[1])[:-1])
        for i, seq in enumerate(seq_list):
            seq_list[i] = seq.reshape(1, -1, no_features)
        return seq_list

    def create_inputs(self, input_array, e_id, fault_start=None):

        if self.no_ins is None:
            return self._create_train_inputs(fault_start, input_array, e_id)
        else:
            return self._create_test_inputs(input_array, e_id)

class PrepRnnInOut_seq:


if __name__ == '__main__':
    from GetCMAPSS import CMAPSS

    raw_data = CMAPSS(1)
    raw_data.get_data()

    selected_feat = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31', 'W32']
    train_df = raw_data.Train_input[selected_feat]
    test_df = raw_data.Test_input[selected_feat]
    e_id_df = raw_data.Train_input['Engine ID']
    e_id_test_df = raw_data.Test_input['Engine ID']
    preper = PrepRnnInOut()

    x, y = preper.create_inputs(train_df.to_numpy(), e_id_df.to_numpy(), np.array([20] * e_id_df.max()))
    z = preper.create_inputs(test_df.to_numpy(), e_id_test_df.to_numpy())
