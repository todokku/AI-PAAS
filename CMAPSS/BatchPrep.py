"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""
import numpy as np


class PrepRnnInOut:

    def __init__(self, faulty_start, e_id, s_len=2, val_split=0.1):
        self.s_len = s_len
        self.val_split = val_split
        self.faulty_start = faulty_start
        self.e_id = e_id

        self.cycle_len = np.unique(e_id, return_counts=True)[-1]
        self.no_ins = np.round((self.cycle_len - self.faulty_start) / self.s_len).astype(int)

    def _assign_dummy(self, x):
        x[x[0] + 1:] = 0.
        return x

    def _prep_inputs(self, input_array):
        engine_seq = np.split(input_array, np.cumsum(np.unique(self.e_id, return_counts=True)[1])[:-1])
        seq_list = []
        for i, seq in enumerate(engine_seq):
            engine_list = []
            for j in range(self.no_ins[i]):
                engine_list.append(np.split(seq, [self.cycle_len[i] - (j + 1) * self.s_len])[0])
            seq_list.extend(engine_list)

        return seq_list

    def _prep_outputs(self):
        outputs = np.arange(self.s_len, self.s_len * (self.no_ins.max() + 1), self.s_len).reshape(-1, 1)
        outputs = np.repeat(outputs, self.e_id.max(), axis=1)
        outputs = np.concatenate((self.no_ins.reshape(1, -1), outputs), axis=0)
        outputs = np.apply_along_axis(self._assign_dummy, 0, outputs)
        outputs = outputs[1:, :]
        outputs = outputs.flatten('F')
        return outputs[outputs != 0.]  # Removing Padded Values

    def create_train_val(self, input_array):

        return self._prep_inputs(input_array), self._prep_outputs()


if __name__ == '__main__':
    from GetCMAPSS import CMAPSS

    raw_data = CMAPSS(1)
    raw_data.get_data()

    selected_feat = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31', 'W32']
    train_df = raw_data.Train_input[selected_feat]

    e_id_df = raw_data.Train_input['Engine ID']

    preper = PrepRnnInOut(np.array([20] * e_id_df.max()), e_id_df.to_numpy())

    x, y = preper.create_train_val(train_df)
