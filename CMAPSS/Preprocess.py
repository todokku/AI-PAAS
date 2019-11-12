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
                 std_fac   = -0.6,    #Std factor
                 s_len     = 2,     #Length of Stagger // Unit - Cycle 
                 pca_var   = 0.97,
                 val_split = 0.4,
                 epsilon   = 1e-5):
        
        self.win_len   = win_len
        self.p_order   = p_order
        self.std_fac   = std_fac
        self.s_len     = s_len
        self.pca_var   = pca_var
        self.val_split = val_split
        self.epsilon   = epsilon
        
# ================================================================================================

    
    def preprocess(self, 
                   input_data,
                   isTrain    = True,
                   force_feat = 0):
        
        self._isTrain    = isTrain
        self._force_feat = force_feat
        self._input_data = input_data
        
        self.no_engines  = self._input_data.iloc[-1,0]
        self._max_cycles = self._input_data['Cycles'].max()
        self._engine_id  = self._input_data.iloc[:,0]
        self._cycles     = self._input_data.iloc[:,1]
        self._input_data = self._input_data.iloc[:,2:]
        
        if self._isTrain:
            self.train_variance = self._input_data.var()
            self._input_data    = self._input_data.loc[:, self.train_variance > self.epsilon]
        else:
            self._input_data = self._input_data.loc[:, self.train_variance > self.epsilon]
        
        if ('Altitude' in 
            self._input_data) or ('Mach Number' in 
                                  self._input_data) or ('TRA' in 
                                                         self._input_data) :
            
            self.clustering()
        
        for i in range(1,self.no_engines+1):
        
            self._input_data.loc[self._engine_id == i,:] = self._input_data.loc[self._engine_id == i,:].apply(self._savgol)
            
        self._input_data = self._input_data.apply(lambda x: (x-x.mean())/x.std())    
        self._input_data = self._input_data.to_numpy()
        
        self.dim_red()
        self.RNN_prep()
                
# ================================================================================================        
    
    def _savgol(self, signal):
    
        smooth_sig = scipy_sig.savgol_filter(signal, 
                                             self.win_len, 
                                             self.p_order, 
                                             mode='nearest')
        return smooth_sig    

# ================================================================================================
    
    def dim_red(self):
        
        if self._isTrain:
        
            pca                 = skl_d.PCA(n_components = self.pca_var, svd_solver ='full')
            self._input_data    = pca.fit_transform(self._input_data)
            self.features = pca.n_components_
        
            print(f'\nNumber of extracted features are {self.features}')
            
        else:
            try:
                features = self.features
            except AttributeError:
                if self._force_feat == 0:
                    raise Exception("Please run train_data first")
                else:
                    features = self._force_feat
                    
            pca = skl_d.PCA(n_components = features)
            pca.fit_transform(self._input_data)
        
            self.test_pca = round(pca.explained_variance_ratio_.sum(),2)
            if(self.test_pca < self.pca_var):
                print(f'PCA test variation is less than the train variation. It is - {self.test_pca}')
            
# ================================================================================================        
  
    def RNN_prep(self):    #Preparing the data for any RNN
        
        if self._isTrain:
                       
            no_ins = np.round(self.no_fcycles/self.s_len)   #fcycles are faulty cycles
            no_ins = no_ins.astype(int)
            
#            total_ins = no_ins.sum()
            
            self.train_out = np.arange(self.s_len*no_ins.max())
            self.train_out = np.repeat(self.train_out,self.no_engines,axis = 1)
            self.train_out = np.concatenate((no_ins.T,self.train_out), axis = 1)
            
            def assign_dummy(x):
                
                x[x[0]:] = 1000
                
                return x
            
            self.train_out = np.apply_along_axis(assign_dummy, 1, self.train_out)
            self.t
            
            self.train_out.flatten()
            idices         = np.arange(self.no_engines*self.s_len*no_ins.max())
            self.train_out = np.delete(self.train_out, idices[self.train_out == 1000])
        
#        self._cycle_len = self._engine_id.value_counts().sort_index().to_numpy()
#        
#        if self._isTrain:
#                       
#            no_ins = np.round(self.no_fcycles/self.s_len)   #fcycles are faulty cycles
#            no_ins = no_ins.astype(int)
#            
#            first_ins = np.append(0, no_ins)
#            first_ins = first_ins.cumsum()        #First Instance of an engine (Used for indexing)
#            
#            total_ins = no_ins.sum()
#            
#            #preparing data for the LSTM
#            self.train_in  = np.full((total_ins, 
#                                      self._max_cycles, 
#                                      self._input_data.shape[1]),
#                                      1000.0)
#                
#            self.train_out = np.full(total_ins, self.epsilon)
#            
#            
#            
#            
#            
#            for i in range(self.no_engines):
#                
#                temp  = self._input_data[self._engine_id == i+1, :]
#                self.train_in[first_ins[i], -self._cycle_len[i]:, :] = temp
#                
#                for j in range(1, no_ins[i]):
#                    
#                    self.train_in [first_ins[i]+j, -self._cycle_len[i]+j*self.s_len:, :] = temp[:-j*self.s_len,:]
#                    self.train_out[first_ins[i]+j] = j*self.s_len
#                    
#        else:
#            self.test_in  = np.full((self.no_engines, 
#                                     self._max_cycles, 
#                                     self._input_data.shape[1]),
#                                     1000.0)
#            
#            for i in range(self.no_engines):
#            
#                c_len = self._cycle_len[i]
#                temp  = self._input_data[self._engine_id == i+1, :]
#                self.test_in[i, -c_len:, :] = temp
                
# ================================================================================================

    def startpt_detection(self):
        
        for i in range(self.no_engines):
    
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

        q25_75 = np.percentile(data,[25,75])
    
        iqr = q25_75[1] - q25_75[0]
    
        data = data[np.all([(st_pts < q25_75[1]+1.5*iqr), (st_pts > q25_75[0]-1.5*iqr)], axis=0) ]
        
        st_pts = outlier(st_pts)
        
        mean[i] = st_pts.mean()
        std[i]  = (st_pts.var())**0.5
        
        self.no_fcycles = 0
    
# ================================================================================================

    def clustering(self):
        pass

# ================================================================================================
            
    def report_card(self):
        
        pass

# ================================================================================================
# ================================================================================================            
          
if __name__ == '__main__':
    
    from Input import cMAPSS as ci
    
#    ci.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')
    ci.get_data(1)
    pp1 = cMAPSS()
    pp1.preprocess(ci.Train_input)
    pp1.preprocess(ci.Test_input, False)

