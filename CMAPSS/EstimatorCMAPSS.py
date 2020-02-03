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
from TestingCMAPSS import Tester


class Estimator:

    def __init__(self, ds_no, window_len=7, poly_order=3, var_threshold=0.9, conf_factor=0, s_len=5, val_split=0,
                 processed_train=False):
        self.conf_factor = conf_factor
        self.s_len = s_len
        self.val_split = val_split
        self.processed_train = processed_train

        self.selected_features = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31', 'W32']

        self.cmapss = CMAPSS(ds_no)
        self.cmapss.get_data()

        if ds_no == 2 or ds_no == 4:
            self.norm = Normalizer(6)
        else:
            self.norm = Normalizer()

        self.de_noise = DeNoiser(window_len, poly_order)
        self.dim_red = DimReduction(var_threshold)


    def get_preprocessed_input(self):

        if self.cmapss.ds_no == 2 or self.cmapss.ds_no == 4:
            op_cond_df = self.cmapss.Train_input.iloc[:, 2:5]
        else:
            op_cond_df = None

        if self.processed_train:
            input_df = self.cmapss.Test_input[self.selected_features]
            e_id_df = self.cmapss.Test_input['Engine ID']
        else:
            input_df = self.cmapss.Test_input[self.selected_features]
            e_id_df = self.cmapss.Test_input['Engine ID']

        input_df = norm.normalising(input_df, op_cond_df)

        input_df = de_noise.smooth(input_df, e_id_df)


        input_array = dim_red.reduce_dimensions(input_df.to_numpy())

        fault_det = FaultDetection(self.conf_factor)
        faulty_cycles = fault_det.get_faulty_cycles(input_df, e_id_df)
        b_prep = PrepRnnInOut(faulty_cycles, e_id_df, s_len, val_split)
        train_tuple = b_prep.create_train_val(train_array)






model_creator = RNNtoFF(dim_red.no_features, [10], [10], epochs=10)
model = model_creator.create_trained_model(train_tuple)

Tester.get_score(model)
