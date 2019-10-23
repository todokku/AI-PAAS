# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:27:17 2019

@author: Tejas
"""

from Input      import cMAPSS as ci
from Preprocess import cMAPSS as CP
import Config as cf

ci.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')

ci.get_data(1)

cp = CP(**cf.preprocess_params)

cp.train_preprocess(ci.Train_input)


