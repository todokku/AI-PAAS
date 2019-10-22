# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:03:55 2019

@author: tejas
"""
#Maybe Create module run to send inputs into
from Input      import cMAPSS as ci
from Preprocess import cMAPSS as CP
import Config as cf

ci.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')

ci.get_data(1)

cp = CP(**cf.sys_params_p,
        **cf.preprocess_params)

cp.train_preprocess(ci.Train_input)

#%%

from   Testing  import cMAPSS as ct
import Training as tr

lstm_ff = tr.LSTM_to_FF(cp.features,
                        **cf.model_hparams,
                        **cf.train_hparams,
                        **cf.sys_params_t)
lstm_ff.create_model()

if cf.sys_params_t['use_gen'] == True:
    lstm_ff.train_model(**cp.npy_files)
else:
    lstm_ff.train_model(train_in = cp.train_in, train_out = cp.train_out)

cp.test_preprocess(ci.Test_input)

ct.get_score(lstm_ff.model, cp.test_in, ci.RUL_input)
