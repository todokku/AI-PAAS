"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""

from GetCMAPSS import CMAPSS
from Normalising import Normalizer
from DeNoising import DeNoiser
from DimensionReduction import DimReduction
from FaultDetection import FaultDetection
from BatchPrep import PrepRnnInOut
from RNNtoFF import RNNtoFF

ds_no = 1

window_len = 7
poly_order = 3

var_threshold = 0.9

conf_factor = 0

s_len = 5
val_split = 0

cmapss = CMAPSS(ds_no)
cmapss.get_data()

selected_feat = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31', 'W32']

if ds_no == 2 or ds_no == 4:
    op_cond_df = cmapss.Train_input.iloc[:, 2:5]
    norm = Normalizer(6)
else:
    op_cond_df = None
    norm = Normalizer()

train_df = cmapss.Train_input[selected_feat]
e_id_df = cmapss.Train_input['Engine ID']

train_df = norm.normalising(train_df, op_cond_df)

de_noise = DeNoiser(window_len, poly_order)
train_df = de_noise.smooth(train_df, e_id_df)

dim_red = DimReduction(var_threshold)
train_array = dim_red.reduce_dimensions(train_df.to_numpy())

fault_det = FaultDetection(conf_factor)
faulty_cycles = fault_det.get_faulty_cycles(train_df, e_id_df)
b_prep = PrepRnnInOut(faulty_cycles, e_id_df, s_len, val_split)
train_tuple = b_prep.create_train_val(train_array)

model_creator = RNNtoFF(dim_red.no_features, [10], [10], epochs=5)
model = model_creator.create_trained_model(train_tuple)

