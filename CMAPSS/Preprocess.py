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
                 std_fac   = -0.6,    #Std factor. Recommended to choose value from -1 to 0
                 s_len     = 2,       #Length of Stagger // Unit - Cycle 
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
        self._e_id  = self._input_data.iloc[:,0]
        self._cycles     = self._input_data.iloc[:,1]
        self._input_data = self._input_data.iloc[:,2:]
        self._cycle_len  = self._e_id.value_counts().sort_index().to_numpy()
        
        
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
            
        self.get_fcycles()
        
        for i in range(1,self.no_engines+1):
        
            self._input_data.loc[self._e_id == i,:] = self._input_data.loc[self._e_id == i,:].apply(self._savgol)
            
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
        
            pca                 = skl_d.PCA(n_components = self.pca_var, 
                                            svd_solver ='full')
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
            no_ins = np.round(no_ins).astype(int).reshape(1,-1)
                     
            self.train_out = np.arange(0, self.s_len*no_ins.max(), self.s_len).reshape(-1,1)  #Generating train_out through vectorising
            self.train_out = np.repeat(self.train_out, self.no_engines, axis = 1)
            self.train_out = np.concatenate((no_ins, self.train_out),   axis = 0)
            self.train_out = np.apply_along_axis(self._assign_dummy, 0, self.train_out)
            self.train_out = self.train_out[1:,:]
            self.train_out = self.train_out.flatten('F')
            
            temp           = self.train_out[self.train_out != 1000]   #Removing Padded Values
            self.train_out = temp
            
            temp = np.repeat(self.no_engines, self._max_cycles)
            
            rem_cycles = temp - self._cycle_len
            
            
            
            
            
            
            
            
            
            idices = self._cycles - 1
            
            cycles_lenlen = np.array([])
            
            for i in range(self.no_engines):
                
                cycles_lenlen = np.append(cycles_lenlen, np.repeat(self._cycle_len[i], self._cycle_len[i]))
                
                
            
            
            
            
            
            self._input_data
            
            
            
            
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

    def _assign_dummy(self, x):
                
        x[x[0]:] = 1000
                
        return x

# ================================================================================================

    def get_fcycles(self): #Provides an estimate for the number of faulty cycles in each engine
        
        def _poly_fit(y):
            
            x = np.arange(len(y))
            
            return np.polyfit(x, y, 2)
        
        fstart_mean     = np.full(self.no_engines, 0)
        fstart_std = np.full(self.no_engines, 0)
        
        for i in range(self.no_engines):
    
            p_coeff = np.apply_along_axis(_poly_fit, 0, self._input_data.loc[self._e_id == i+1,:])
            
            temp = np.all([-p_coeff[1, :]/(2*p_coeff[0, :]) > 1, 
                           -p_coeff[1, :]/(2*p_coeff[0, :]) < self._cycle_len[i]], 
                           axis = 0)
            
            temp    = p_coeff[:, temp]
            p_coeff = temp
            st_pts  = np.apply_along_axis(np.roots, 0, p_coeff[:2,:])
                    
            q25_75 = np.percentile(st_pts,[25,75])
    
            iqr  = q25_75[1] - q25_75[0]
            temp = st_pts[np.all([(st_pts < q25_75[1]+1.5*iqr), 
                                  (st_pts > q25_75[0]-1.5*iqr)], 
                                  axis=0)]
            st_pts = temp
            
            fstart_mean[i] = st_pts.mean()
            fstart_std[i]  = (st_pts.var())**0.5
        
        self.fstart = fstart_mean + fstart_std*self.std_fac
        
        self.no_fcycles = self._cycle_len - self.fstart
        self.no_fcycles = np.round(self.no_fcycles).astype(int)
         
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
    
    ci.set_datapath('C:/Users/Tejas/Desktop/Tejas/engine-dataset/')
    ci.get_data(1)
    pp1 = cMAPSS()
    pp1.preprocess(ci.Train_input)
    pp1.preprocess(ci.Test_input, False)

