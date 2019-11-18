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
 
