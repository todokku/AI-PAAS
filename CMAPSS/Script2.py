from GetCMAPSS import CMAPSS
from Normalising import Normalizer
from SequencePrep import PrepRnnInOutSeq
import matplotlib.pyplot as plt
from RNNtoFF import RNNtoFFSeq
from FaultDetection import FaultDetector
# %%

ds_no = 4
cmapss = CMAPSS(ds_no)
cmapss.get_data()
selected_feat = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31', 'W32']
train_df = cmapss.Train_input[selected_feat]
e_id_df = cmapss.Train_input['Engine ID']

if ds_no == 2 or ds_no == 4:
    op_cond_df = cmapss.Train_input.iloc[:, 2:5]
    normalizer = Normalizer(6)
else:
    op_cond_df = None
    normalizer = Normalizer(1)


f_detec = FaultDetector(0)

train_df = normalizer.normalise(train_df, op_cond_df)

fault_start = f_detec.get_fault_start(train_df, e_id_df)

input_seq = PrepRnnInOutSeq()._prep_train_inputs(train_df.to_numpy(), e_id_df.to_numpy())

output_seq = PrepRnnInOutSeq()._prep_train_outputs(fault_start, e_id_df)

for i in range(2):
    rnntoff = RNNtoFFSeq(input_seq[0].shape[2], [5], [], 'simpleRNN', epochs=1, dropout=0.4, rec_dropout=0.2,
                         final_activation='relu')
    rnntoff.create_trained_model((input_seq, output_seq))