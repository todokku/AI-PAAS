# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:16:00 2019

@author: tejas
"""

from   Input       import cMAPSS     as ci
from   Preprocess  import cMAPSS     as cp 
from   Training    import LSTM_to_FF as lstm_ff  
from   Testing     import cMAPSS     as ct

#ci.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')

ci.get_data(1)

cp.train_preprocess(ci.Train_input)

#%%

lstm_ff.config_model(1, 10, 1, 2)
lstm_ff.create_model([cp.max_cycles, cp.train_in.shape[2]])
lstm_ff.model_train(cp.train_in,cp.train_out)


cp.test_preprocess(ci.Test_input)

ct.get_score(lstm_ff.model, cp.test_in, ci.RUL_input)
