# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:16:00 2019

@author: tejas
"""

from   Input       import cMAPSS as ci
from   Preprocess  import cMAPSS as cp   

#import numpy as np

ci.get_data(1)

cp.preprocess(ci.Train_input)

#%%
