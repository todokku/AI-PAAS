# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:03:55 2019

@author: tejas
"""

from Input      import cMAPSS as ci
from Preprocess import cMAPSS as CP
from Testing    import cMAPSS as ct

import Training as tr



ci.get_data(1)

cp = CP()
cp.train_preprocess(ci.Train_input)

lstm_ff = tr.LSTM_to_FF(cp.features,
                        lstm_layer   = 3, 
                        lstm_neurons = 300, 
                        ff_layer     = 2, 
                        ff_neurons   = 150,
                        epochs       = 40)

lstm_ff.create_model()
#lstm_ff.train_model(cp.tin_npy,cp.tout_npy,cp.vin_npy,cp.vout_npy)

lstm_ff.train_model(train_in = cp.train_in, train_out = cp.train_out)


cp.test_preprocess(ci.Test_input)

ct.get_score(lstm_ff.model, cp.test_in, ci.RUL_input)
