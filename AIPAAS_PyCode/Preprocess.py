# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student

"""

# =================================================================================================
# # Always use s_rep>1 to give non similar train_out values
# =================================================================================================

#Libraries       This module needs optimising , use iterators
#uSe single module config files
import scipy.signal          as scipy_sig
import numpy                 as np
import sklearn.decomposition as skl_d
import os

class cMAPSS:

    def __init__(self, 
                 win_len   = 21, 
                 p_order   = 3, 
                 threshold = 1e-5, 
                 s_rep     = 2,    #Stagered Repetition
                 s_len     = 60,   #Unit - Cycle 
                 pca_var   = 0.90,
                 val_split = 0.4,
                 use_gen   = False):
        
        self.win_len   = win_len
        self.p_order   = p_order
        self.threshold = threshold
        self.s_rep     = s_rep
        self.s_len     = s_len
        self.pca_var   = pca_var
        self.val_split = val_split
        self.use_gen   = use_gen
    
    def savgol(self, signal):
    
        smooth_sig = scipy_sig.savgol_filter(signal, 
                                             self.win_len, 
                                             self.p_order, 
                                             mode='nearest')
        return smooth_sig

    def basic_preprocess(self, input_data):
        
        self.no_engines = input_data.iloc[-1,0]
        self.max_cycles = input_data['Cycles'].max()
        engine_id       = input_data.iloc[:,0]
        input_data      = input_data.iloc[:,2:]

        data_variance   = input_data.var()
        input_data      = input_data.loc[:, data_variance > self.threshold]
    
        for i in range(1,self.no_engines+1):
        
            input_data.loc[engine_id == i,:] = input_data.loc[engine_id == i,:].apply(self.savgol)
            
        #Normalising after data has been filtered
        input_data = input_data.apply(lambda x: (x-x.mean())/x.std())
                
        return engine_id, input_data
    
    def train_preprocess(self, train_data):
        
        engine_id, train_data = self.basic_preprocess(train_data)
                
        pca           = skl_d.PCA(n_components = self.pca_var, svd_solver ='full')
        train_data    = pca.fit_transform(train_data)
        self.features = pca.n_components_
        
        print(f'\nNumber of extracted features are {self.features}')
        
        #preparing data for the LSTM
        self.train_in  = np.full((self.no_engines*self.s_rep, 
                                 self.max_cycles, 
                                 train_data.shape[1]),
                                 1000.0)
            
        self.train_out = np.full((self.no_engines*self.s_rep),1e-2)
        
        for i in range(0, self.no_engines*self.s_rep, self.s_rep):
            
            e_id      = i/self.s_rep + 1
            cycle_len = train_data[engine_id == e_id, :].shape[0]
            temp      = train_data[engine_id == e_id, :]
            self.train_in[i, :cycle_len, :] = temp
            
            for j in range(1, self.s_rep):
                
                self.train_in[j+i, :cycle_len-self.s_len*j, :] = temp[:-self.s_len*j,:]
                self.train_out[j+i] = self.s_len*j
                
        train_id = np.arange(self.no_engines*self.s_rep)
        np.random.shuffle(train_id)
        
        tv_s     = int(np.ceil(self.val_split*self.no_engines*self.s_rep))
        val_id   = train_id[ : tv_s]
        train_id = train_id[tv_s : ]
        
        if self.mp == True :
            
            self.tin_npy  = './np_cache/tin_data.npy'
            self.tout_npy = './np_cache/tout_data.npy'
            self.vin_npy  = './np_cache/vin_data.npy'
            self.vout_npy = './np_cache/vout_data.npy'
            
            if os.path.isfile(self.tin_npy):
                
                os.remove(self.tin_npy)
                os.remove(self.tout_npy)
                os.remove(self.vin_npy)
                os.remove(self.vout_npy)
            
            np.save(self.tin_npy ,self.train_in [train_id,:,:])
            np.save(self.tout_npy,self.train_out[train_id])
            np.save(self.vin_npy ,self.train_in [val_id  ,:,:])
            np.save(self.vout_npy,self.train_out[val_id])
                
    def test_preprocess(self, test_data):
        
        engine_id, test_data = self.basic_preprocess(test_data)
        
        try:
            pca = skl_d.PCA(n_components = self.features)
        except AttributeError: 
            raise Exception("Please run train_data first")
        
        pca.fit(test_data)
        
        if(pca.explained_variance_ratio_.sum() < self.pca_var*0.98):
            raise Exception("Ummmm.... There is an issue with the pca thing, call - 289 923 9291")
        
        test_data   = pca.transform(test_data)
        
        #preparing data for the LSTM
        self.test_in  = np.full((self.no_engines, 
                                 self.max_cycles, 
                                 test_data.shape[1]),
                                 1000.0)
            
        for i in range(self.no_engines):
            
            cycle_len = test_data[engine_id == i+1, :].shape[0]
            temp      = test_data[engine_id == i+1, :]
            self.test_in[i, :cycle_len, :] = temp
            
  
if __name__ == '__main__':
    
    from Input import cMAPSS as ci
    
    ci.get_data(1)
    pp1 = cMAPSS()
    pp1.train_preprocess(ci.Train_input)
    pp1.test_preprocess(ci.Test_input)

