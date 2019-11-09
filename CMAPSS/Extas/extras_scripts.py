# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 01:29:45 2019

@author: tejas
"""

from Input      import cMAPSS as ci
from Preprocess import cMAPSS as CP
import os
import numpy as np

ci.get_data(1)

cp = CP()
cp.train_preprocess(ci.Train_input)

a = cp.train_in

train_id = np.arange(200)
np.random.shuffle(train_id)

tv_s     = int(np.ceil(0.4*200))
val_id   = train_id[ : tv_s]
train_id = train_id[tv_s : ]
        
tin_npy  = './np_cache/tin_data.npy'
tout_npy = './np_cache/tout_data.npy'

if os.path.isfile(tin_npy):
   os.remove(tin_npy)
   os.remove(tout_npy)

b = cp.train_in [train_id,:,:]
c = cp.train_out[train_id]
np.save(tin_npy , b)
np.save(tout_npy , c)

data_in  = np.load(tin_npy, mmap_mode = 'r')
data_out = np.load(tout_npy, mmap_mode = 'r')

batch_size = 32

batch_no = int(np.ceil(120/batch_size))

def getitem(index, batch_size, batch_no):
        
    if index == batch_no-1:
        return data_in[index*batch_size: , : , : ], data_out[index*batch_size: ]
    else:
        return data_in[index*batch_size : (index+1)*batch_size,:,:], data_out[index*batch_size:(index+1)*batch_size ] 
        



