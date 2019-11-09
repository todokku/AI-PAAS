# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:46:15 2019

@author: Tejas
"""

import Testing as test

import Config



from Preprocess import cMAPSS as C
from Input import cMAPSS as i

model = test.read_model('../KerasModels/23_Oct_19__14_04__158_108.hdf5')

print(model.summary())

i.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')

i.get_data(1)

c = C(**Config.prepros_params)

c.test_preprocess(i.Test_input,2)

test.cMAPSS.get_score(model, c.test_in, i.RUL_input)
