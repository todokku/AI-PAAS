# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:27:17 2019


@author: Tejas
"""

from Input      import cMAPSS as ci
import numpy as np
#from Preprocess import cMAPSS as CP
#import Config as cf

#ci.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')

ci.get_data(1)

a = ci.RUL_input

min_ = ci.RUL_input.min().to_numpy()
max_ = ci.RUL_input.max().to_numpy()

c = np.random.randint(min_, max_, len(ci.RUL_input))

#cp = CP(**cf.preprocess_params)
#
#cp.train_preprocess(ci.Train_input)

d=c.copy()
rmse = (d**2)
rmse = (rmse.mean())**0.5
        
d[d>=0]  = np.exp(d[d>=0]/10) - 1
d[d<0]   = np.exp(-(d[d<0]/13)) - 1
s    = int(np.round(d.sum()))