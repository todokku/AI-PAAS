# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 01:29:45 2019

@author: tejas
"""

#import Config as c
#
#import Run
#
#Run.cMAPPS(c.prepros_params,
#           c.train_params,
#           1,
#           path = 'C:/Users/Tejas/Desktop/Tejas/engine-dataset/')




import numpy as np

import tensorflow.keras as tfk


#Create input sequence


x = np.full(40,0)


x[20:] = np.arange(1, 21)

x=x.reshape(-1,40,1)

y = np.array([54])

model = tfk.models.Sequential([tfk.layers.Masking(mask_value = 50, input_shape=(None, 1)),
                               tfk.layers.SimpleRNN(1)])
model.compile(loss = 'mse',
              optimizer = 'adam')
#print(model.summary())

model.fit(x,y)

x = np.arange(1, 21)

x=x.reshape(-1,20,1)

y = np.array([54])

model = tfk.models.Sequential([tfk.layers.SimpleRNN(1, input_shape=(None, 1))])
model.compile(loss = 'mse',
              optimizer = 'adam')
#print(model.summary())

model.fit(x,y)


