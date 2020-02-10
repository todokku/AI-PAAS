"""
AI-AS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-AS Phd Student

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

    def __init__(self, ds_no, window_len=7, poly_order=3, var_threshold=0.9, conf_factor=0, s_len=5,
                 initial_cutoff=0.75, ins_dropped=0.25, rnn_neurons=[10, 10], ff_neurons=[10], rnn_type='simpleRNN',
                 epochs=1, lRELU_alpha=0.05, lr=0.001, dropout=0.4, rec_dropout=0.2, l2_k=0.001, l2_b=0., l2_r=0.,
                 model_dir=None, run_id=None, enable_norm=True, final_activation=None):
        self.processed_train = False
        self.selected_features = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31',
                                  'W32']
        self.cmapss = CMAPSS(ds_no)
        self.cmapss.get_data()

        if ds_no == 2 or ds_no == 4:
            self.norm = Normalizer(6)
        else:
            self.norm = Normalizer()

        self.de_noise = DeNoiser(window_len, poly_order)
        self.dim_red = DimReduction(var_threshold)
        self.fault_det = FaultDetection(conf_factor)
        self.batch_prep = PrepRnnInOut(s_len, initial_cutoff, ins_dropped)
        self.tester = Tester()

        self.model_manager = None
        self.train_tuple = None
        self.model = None

        self.rnn_neurons = rnn_neurons
        self.ff_neurons = ff_neurons
        self.rnn_type = rnn_type
        self.epochs = epochs
        self.lRELU_alpha = lRELU_alpha
        self.lr = lr
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.l2_k = l2_k
        self.l2_b = l2_b
        self.l2_r = l2_r
        self.run_id = run_id
        self.model_dir = model_dir
        self.enable_norm = enable_norm
        self.final_activation = final_activation

    def _get_preprocessed_input(self):

        if self.cmapss.ds_no == 2 or self.cmapss.ds_no == 4:
            op_cond_df = self.cmapss.Train_input.iloc[:, 2:5]
        else:
            op_cond_df = None

        if self.processed_train:
            input_df = self.cmapss.Test_input[self.selected_features]
            e_id_df = self.cmapss.Test_input['Engine ID']
        else:
            input_df = self.cmapss.Train_input[self.selected_features]
            e_id_df = self.cmapss.Train_input['Engine ID']

        input_df = self.norm.normalising(input_df, op_cond_df)
        input_df = self.de_noise.smooth(input_df, e_id_df)
        input_array = self.dim_red.reduce_dimensions(input_df.to_numpy())
        if not self.processed_train:
            self.model_manager = RNNtoFF(self.dim_red.no_features, self.rnn_neurons, self.ff_neurons, self.rnn_type,
                                         self.epochs, self.lRELU_alpha, self.lr, self.dropout, self.rec_dropout,
                                         self.l2_k, self.l2_b, self.l2_r, self.run_id, self.model_dir,
                                         enable_norm=self.enable_norm, final_activation=self.final_activation)
            self.processed_train = True

        return self.batch_prep.create_inputs(input_array, e_id_df.to_numpy(), self.fault_det.get_fault_start(input_df,
                                                                                                             e_id_df))

    def evaluate_params(self):
        self._create_trained_model()
        self._test_model()

    def _create_trained_model(self):
        self.train_tuple = self._get_preprocessed_input()

        self.model = self.model_manager.create_trained_model(self.train_tuple)

    def retrain_model(self):

        self.model = self.model_manager.retrain_model(self.model, self.train_tuple)

    def _test_model(self):
        if self.model is None:
            raise Exception('Create Model First')
        self.score = self.tester.get_score(self.model, self._get_preprocessed_input(), self.cmapss.RUL_input.to_numpy())


if __name__ == '__main__':
    estimator = Estimator(1)

    estimator.evaluate_params()
