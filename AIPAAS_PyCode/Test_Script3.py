# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:46:15 2019

@author: Tejas
"""

import Testing as test

import Config



from Preprocess import cMAPSS as C
from Input import cMAPSS as i

i.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')

i.get_data(1)

c = C(**Config.preprocess_params)

c.test_preprocess(i.Test_input,3)

test.cMAPSS.get_score(test.read_model('../KerasModels/22_Oct_19__14_43__89_38.hdf5'), c.test_in, i.RUL_input)
