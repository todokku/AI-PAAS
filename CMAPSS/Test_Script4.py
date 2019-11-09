# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 01:41:56 2019

@author: tejas
"""

import numpy as np

import HyperParamTuners as hpt

def ackley(x,y,z):
    
    a = 20
    b = 0.2
    c = 2*np.pi
    
    x = np.array([x,y,z])
    
#    d = -a*np.exp(-b*(np.sum(x**2, axis = 1)/x.shape[0])**0.5)
#    e = -np.exp(np.sum(np.cos(c*x), axis = 1)/x.shape[0])
    
    d = -a*np.exp(-b*(np.sum(x**2)/x.shape[0])**0.5)
    e = -np.exp(np.sum(np.cos(c*x)/x.shape[0]))
    
    return -(d+e+a+np.exp(1))


#y = ackley(np.array([[3,2],[0,0]]))


bopt = hpt.BayesOpt(ackley,
                    {'x' : (-32.768, 32.768),
                     'y' : (-32.768, 32.768),
                     'z' : (-32.768, 32.768)})