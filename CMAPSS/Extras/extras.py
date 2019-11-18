# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:37:41 2019

@author: tejas
"""

def generator_preprocess(self):
    
    train_id = np.arange(self.no_engines*self.s_rep)
    np.random.shuffle(train_id)
    
    tv_s     = int(np.ceil(self.val_split*self.no_engines*self.s_rep))
    val_id   = train_id[ : tv_s]
    train_id = train_id[tv_s : ]
    
    tin_npy  = './np_cache/tin_data.npy'
    tout_npy = './np_cache/tout_data.npy'
    vin_npy  = './np_cache/vin_data.npy'
    vout_npy = './np_cache/vout_data.npy'
        
    if os.path.isfile(tin_npy):
            
        os.remove(tin_npy)
        os.remove(tout_npy)
        os.remove(vin_npy)
        os.remove(vout_npy)
        
    np.save(tin_npy ,self.train_in [train_id,:,:])
    np.save(tout_npy,self.train_out[train_id])
    np.save(vin_npy ,self.train_in [val_id  ,:,:])
    np.save(vout_npy,self.train_out[val_id])
        
    self.npy_files = {'tin_npy'  : './np_cache/tin_data.npy',
                      'tout_npy' : './np_cache/tout_data.npy',
                      'vin_npy'  : './np_cache/vin_data.npy',
                      'vout_npy' : './np_cache/vout_data.npy'}
        
    del self.train_in
    del self.train_out
    
    
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, in_npy, out_npy, batch_size): 
        
        self.input_data  = np.load(in_npy,  mmap_mode = 'r')     #input output w.r.t the final model
        self.output_data = np.load(out_npy, mmap_mode = 'r') 
        self.batch_size  = batch_size
        self.batch_no    = int(np.ceil(self.input_data.shape[0] / self.batch_size))
    
    def __len__(self):
        return self.batch_no
    
    def __getitem__(self, index):
        
        if index == self.batch_no-1:
            
            X = np.copy(self.input_data[index*self.batch_size: , : , : ])
            y = np.copy(self.output_data[index*self.batch_size: ])
            
            return X,y
        
        else:
            X = np.copy(self.input_data[index*self.batch_size:(index+1)*self.batch_size , : , : ])
            y = np.copy(self.output_data[index*self.batch_size: (index+1)*self.batch_size])
            
            return X, y
        
    def on_epoch_end(self):
        
       pass
   
    
    
    
#preprocess vectorization
       

def _assign_dummy(self, x):
            
    x[x[0]:] = 1000
            
    return x
        
if self._isTrain:
    
    self._no_ins = np.round(self.no_fcycles/self.s_len)   #fcycles are faulty cycles
    self._no_ins = np.round(self._no_ins).astype(int).reshape(1,-1)
             
    self.train_out = np.arange(0, self.s_len*self._no_ins.max(), self.s_len).reshape(-1,1)  #Generating train_out through vectorising
    self.train_out = np.repeat(self.train_out, self.no_engines, axis = 1)
    self.train_out = np.concatenate((self._no_ins, self.train_out),   axis = 0)
    self.train_out = np.apply_along_axis(self._assign_dummy, 0, self.train_out)
    self.train_out = self.train_out[1:,:]
    self.train_out = self.train_out.flatten('F')
    
    temp           = self.train_out[self.train_out != 1000]   #Removing Padded Values
    self.train_out = temp
    
    rem_cycles = np.repeat(self._max_cycles, self.no_engines)
    rem_cycles = rem_cycles - self._cycle_len
    
    index = np.arange(0, self._max_cycles*self.no_engines, self._max_cycles)
    
    self.train_in = self._input_data
    
    for i in range(self.no_engines):

        self.train_in = np.concatenate((self.train_in[:index[i] , :], 
                                        np.full((rem_cycles[i],self.features), 1000), 
                                        self.train_in[index[i]: , :]), 
                                       axis = 0)  
        
    self.train_in = np.repeat(self.train_in[np.newaxis, :, :], self._no_ins.max(), axis = 0) 
    
    self.train_in = self.train_in(-1, self.features, self._max_cycles)
    
    temp = np.arange(self._np_ins.max())
    
    def stagger
    
    self.train_in = self.train_in.reshape(self._max_cycles, self.features, -1)
 
