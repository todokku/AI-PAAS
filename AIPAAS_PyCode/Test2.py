# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:16:00 2019

@author: tejas
"""

from   Input       import cMAPSS     as ci
from   Preprocess  import cMAPSS     as CP
from   Training    import LSTM_to_FF 
from   Testing     import cMAPSS     as ct

ci.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')

ci.get_data(1)

cp = CP()
cp.train_preprocess(ci.Train_input)

#%%

lstm_ff = LSTM_to_FF(cp.train_in.shape[2],
                     lstm_layer   = 3,
                     ff_layer     = 2,
                     lstm_neurons = 500,
                     ff_neurons   = 500,
                     epochs       = 20)

#lstm_ff = LSTM_to_FF(cp.train_in.shape[2],
#                     lstm_layer   = 1,
#                     ff_layer     = 1,
#                     lstm_neurons = 3,
#                     ff_neurons   = 1)

lstm_ff.create_model()
lstm_ff.train_model(cp.train_in,cp.train_out)

cp.test_preprocess(ci.Test_input)

ct.get_score(lstm_ff.model, cp.test_in, ci.RUL_input)
