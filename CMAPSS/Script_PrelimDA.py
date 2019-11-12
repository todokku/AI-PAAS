# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:57:26 2019

@author: tejas
"""

from Input import cMAPSS as ci
import scipy.signal      as scipy_sig
#import matplotlib.pyplot as plt
import numpy             as np
import scipy.stats       as scipy_stat

def savgol(signal):
    
    smooth_sig = scipy_sig.savgol_filter(signal, 
                                         51, 
                                         2, 
                                         mode='nearest')
    return smooth_sig

def exp_coeff(x, a, b):
    return np.exp(x*a) + b

ci.get_data(1)
Train_Input = ci.Train_input
Test_Input  = ci.Test_input

no_engines = Train_Input.iloc[-1,0]
e_id       = Train_Input.iloc[:,0]
cycles     = Train_Input.iloc[:,1]
input_data = Train_Input.iloc[:,2:]
test_data  = Test_Input.iloc[:,2:]

eng = [0]*no_engines
cyc = [0]*no_engines

data_variance   = input_data.var()
input_data      = input_data.loc[:, data_variance > 1e-5]
data_variance   = test_data.var()
test_data       = test_data.loc[:, data_variance > 1e-5]

mean = np.full(no_engines,0.0)
std  = np.full(no_engines,0.0) 
#starting_points = []
#des = []
#outlier_det = sk_svm.OneClassSVM(gamma='scale')

def outlier(data):
    
    q25_75 = np.percentile(data,[25,75])
    
    iqr = q25_75[1] - q25_75[0]
    
    data = data[np.all([(st_pts < q25_75[1]+1.5*iqr), (st_pts > q25_75[0]-1.5*iqr)], axis=0) ]

    return data

for i in range(no_engines):
    
    cyc[i] = cycles.loc[e_id == i+1]
    
    eng[i] = input_data.loc[e_id == i+1,:]
    eng[i] = eng[i].set_index(cyc[i])
    st_pts = np.array([])
    for j in range(input_data.shape[1]):
        
        
        
        p_coeff = np.polyfit(cyc[i], 
                             eng[i].iloc[:,j],
                             2)
    
        if -p_coeff[1]/(2*p_coeff[0]) > cyc[i].iloc[0] and -p_coeff[1]/(2*p_coeff[0]) < cyc[i].iloc[-1]:
            
            st_pt = np.roots(p_coeff[:2]).max()
            
            st_pts = np.append(st_pts, st_pt)

    st_pts = outlier(st_pts)
    
    mean[i] = st_pts.mean()
    std[i]  = (st_pts.var())**0.5

cycle_len = e_id.value_counts().sort_index().to_numpy()

st_pts = mean - 0.6*std

describe_sp = scipy_stat.describe(st_pts)

per = 100*(1-st_pts/cycle_len)

#%%
#
#n=6
#
#for i in range(input_data.shape[1]):
#   
#    p_coeff = np.polyfit(cyc[n], 
#                         eng[n].iloc[:,i],
#                         2)
#
#    plt.figure(i)
#  
#    eng[n].iloc[:,i].plot()   
#    p = np.poly1d(p_coeff)
#    
#    plt.plot(cycles.loc[e_id == n+1],p(cycles.loc[e_id == n+1]))
#    
#    if not(-p_coeff[1]/(2*p_coeff[0]) > cyc[i].iloc[0] and -p_coeff[1]/(2*p_coeff[0]) < cyc[i].iloc[-1]):
#            
#        print(f'{i} is the issue')
#    
#    
#
#


