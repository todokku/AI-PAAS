# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:27:22 2019

@author: tejas
"""

from Input      import cMAPSS as ci

ci.get_data(1)
a = ci.Train_input
b = ci.RUL_input


a = a.iloc[: , 2: ]

var = a.var()

a = a.loc[:,var>10e-5]

corr = a.corr()
