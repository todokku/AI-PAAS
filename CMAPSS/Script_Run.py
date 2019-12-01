# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:03:55 2019

@author: tejas
"""
#Maybe Create module run to send inputs into
from   Input      import cMAPSS as ci
from   Preprocess import cMAPSS as CP
import Config as cf

#ci.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')

ci.get_data(1)

cp = CP(**cf.prepros_params)

cp.preprocess(ci.Train_input)

#%%

import Training as tr
from   Testing import cMAPSS as ct

Model = tr.RNN_to_FF(cp.features,
                      **cf.train_params)
Model.create_model()

Model.train_model(cp.train_in, cp.train_out, cp.val_in, cp.val_out)

Model.history_plot()


cp.preprocess(ci.Test_input,
              isTrain = False)

ct.get_score(Model.model, cp.test_in, ci.RUL_input)


