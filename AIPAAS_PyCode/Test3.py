# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:27:22 2019

@author: tejas
"""

from Input      import cMAPSS as ci
from sklearn.decomposition import PCA
import scipy.signal      as scipy_sig

ci.get_data(1)
a = ci.Train_input
b = ci.RUL_input

e_id = a.iloc[:,0]
a = a.iloc[: , 2: ]

var = a.var()

a = a.loc[:,var>10e-5]
#
#corr = a.corr()
#
#a = a.apply(lambda x: (x-x.mean())/(x.max()-x.min()))
a = a.apply(lambda x: (x-x.mean())/x.std())
pca = PCA(n_components=0.9, svd_solver ='full')
 
pca.fit(a) 
c=pca.explained_variance_ratio_
e=pca.explained_variance_
d=pca.singular_values_
g=c.cumsum()

h=pca.transform(a)
#def savgol(signal):
#    
#        smooth_sig = scipy_sig.savgol_filter(signal, 
#                                             21, 
#                                             3, 
#                                             mode='nearest')
#        return smooth_sig
#    
#a = a.apply(lambda x: (x-x.mean())/x.std())
#
#for i in range(1,101):
#            
##            temp.append(input_data.loc[engine_id == i,:].apply(cls.savgol))
#        
#    a.loc[e_id == i,:] = a.loc[e_id == i,:].apply(savgol)


pca.n_components_







