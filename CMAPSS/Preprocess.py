# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student

"""

# =================================================================================================
# Preprocessing Module!
# =================================================================================================

# TODO        This module needs optimising , use np.apply over axis

#Libraries

import scipy.signal          as scipy_sig
import numpy                 as np
import sklearn.decomposition as skl_d

class cMAPSS:

    def __init__(self, 
                 win_len   = 21, 
                 p_order   = 3, 
                 s_per     = 35,    #Stagered Percentage
                 s_len     = 5,     #Length of Stagger // Unit - Cycle 
                 pca_var   = 0.97,
                 val_split = 0.4,
                 epsilon   = 1e-5):
        
        self.win_len   = win_len
        self.p_order   = p_order
        self.s_per     = s_per
        self.s_len     = s_len
        self.pca_var   = pca_var
        self.val_split = val_split
        self.epsilon   = epsilon
    
    def savgol(self, signal):
    
        smooth_sig = scipy_sig.savgol_filter(signal, 
                                             self.win_len, 
                                             self.p_order, 
                                             mode='nearest')
        return smooth_sig
    
    def clustering(self):
        pass

    def basic_preprocess(self, input_data):
        
        self.no_engines = input_data.iloc[-1,0]
        self.max_cycles = input_data['Cycles'].max()
        engine_id       = input_data.iloc[:,0]
        input_data      = input_data.iloc[:,2:]

        data_variance   = input_data.var()
        input_data      = input_data.loc[:, data_variance > self.epsilon]
        
        if ('Altitude' in input_data) or ('Mach Number' in input_data) or ('TRA' in input_data) :
            
            self.clustering()
        
        cycle_len = np.full(self.no_engines,0)
        for i in range(1,self.no_engines+1):
        
            input_data.loc[engine_id == i,:] = input_data.loc[engine_id == i,:].apply(self.savgol)
            cycle_len[i-1] = len(engine_id[engine_id == i])
            
        #Normalising after data has been filtered
        input_data = input_data.apply(lambda x: (x-x.mean())/x.std())
                
        return cycle_len, engine_id, input_data
    
    def train_preprocess(self, train_data):
        
        cycle_len, engine_id, train_data = self.basic_preprocess(train_data)
                
        pca           = skl_d.PCA(n_components = self.pca_var, svd_solver ='full')
        train_data    = pca.fit_transform(train_data)
        self.features = pca.n_components_
        
        print(f'\nNumber of extracted features are {self.features}')
            
        self.no_ins = np.round(cycle_len*self.s_per/100)
        self.no_ins = np.round(self.no_ins/5)
        self.no_ins = self.no_ins.astype(int)
        
        first_ins = np.append(0, self.no_ins)
        first_ins = first_ins.cumsum()
        
        no_engine_ins = self.no_ins.sum()
        
        #preparing data for the LSTM
        self.train_in  = np.full((no_engine_ins, 
                                  self.max_cycles, 
                                  train_data.shape[1]),
                                  1000.0)
            
        self.train_out = np.full((no_engine_ins), self.epsilon)
        
        for i in range(self.no_engines):
            
            c_len = cycle_len[i]
            temp  = train_data[engine_id == i+1, :]
            self.train_in[first_ins[i], -c_len:, :] = temp
            
            for j in range(1, self.no_ins[i]):
                
#                self.train_in [first_ins[i]+j, -c_len:-j*self.s_len, :] = temp[:-j*self.s_len,:]
                
                self.train_in [first_ins[i]+j, -c_len+j*self.s_len:, :] = temp[:-j*self.s_len,:]
                self.train_out[first_ins[i]+j] = j*self.s_len
                
    def test_preprocess(self, test_data, feat = 0):
        
        try:
            features = self.features
        except AttributeError:
            if feat == 0:
                raise Exception("Please run train_data first")
            else:
                features = feat

        cycle_len, engine_id, test_data = self.basic_preprocess(test_data)
        
        pca = skl_d.PCA(n_components = features)
        pca.fit(test_data)
        
        if(round(pca.explained_variance_ratio_.sum(),2) < round(self.pca_var,2)):
            print(f'PCA test variation is less than the train variation. It is - {self.pca_var}')
        
        test_data   = pca.transform(test_data)
        
        #preparing data for the LSTM
        self.test_in  = np.full((self.no_engines, 
                                 self.max_cycles, 
                                 test_data.shape[1]),
                                 1000.0)
            
        for i in range(self.no_engines):
            
            c_len = cycle_len[i]
            temp  = test_data[engine_id == i+1, :]
            self.test_in[i, -c_len:, :] = temp
            
          
if __name__ == '__main__':
    
    from Input import cMAPSS as ci
    
#    ci.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')
    ci.get_data(1)
    pp1 = cMAPSS()
    pp1.train_preprocess(ci.Train_input)
    pp1.test_preprocess(ci.Test_input)

