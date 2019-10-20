# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student

"""

# =========================================================================================
# Choose an appropriate model to create and Train
# =========================================================================================

import tensorflow        as tf
import matplotlib.pyplot as plt
import numpy             as np
import datetime

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
   

class LSTM_to_FF:
        
    def __init__(self,
                 features,
                 lstm_layer   = 1, 
                 lstm_neurons = 100, 
                 ff_layer     = 1, 
                 ff_neurons   = 100,
                 epochs       = 10,
                 val_split    = 0.4,
                 batch_size   = 64,
                 use_gen      = False):
        
        self.lstm_layer = lstm_layer 
        self.ff_layer   = ff_layer
    
        self.lstm_neurons = lstm_neurons
        self.ff_neurons   = ff_neurons
        
        self.features  = features
        self.epochs    = epochs 
        self.val_split = val_split
        self.mp        = mp
        
        self.batch_size = 32
        
# ================================================================================================
    
    def create_model(self):
        
        #add batch normalization
        self.model = tf.keras.models.Sequential()
        
        self.model.add(tf.keras.layers.Masking(mask_value = 1000.0 ,
                                               input_shape=(None,self.features)))
        
        for i in range(0, self.lstm_layer-1):
            
            self.model.add(tf.keras.layers.LSTM(self.lstm_neurons, return_sequences=True))
            self.model.add(tf.keras.layers.Dropout(0.4))
            self.model.add(tf.keras.layers.ActivityRegularization(l2=0.001))
    
        self.model.add(tf.keras.layers.LSTM(self.lstm_neurons))
        self.model.add(tf.keras.layers.Dropout(0.4))
        self.model.add(tf.keras.layers.ActivityRegularization(l2=0.001))
        
        for i in range(0, self.ff_layer):
               
            self.model.add(tf.keras.layers.Dense(self.ff_neurons))
            self.model.add(tf.keras.layers.Dropout(0.4))
            self.model.add(tf.keras.layers.ActivityRegularization(l2=0.001))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
       
        #Final Layer
        self.model.add(tf.keras.layers.Dense(1))
        
        self.model.compile(loss='mse',
                          optimizer='adam')
        
        print(self.model.summary())
        
# ================================================================================================
        
    def train_model(self, 
                    tin_npy   = None, 
                    tout_npy  = None,
                    vin_npy   = None,
                    vout_npy  = None,
                    train_in  = None, 
                    train_out = None):
        
        if self.mp == True:
            
            train_gen = DataGenerator(tin_npy, tout_npy, self.batch_size)
            val_gen   = DataGenerator(vin_npy, vout_npy, self.batch_size)
            #TODO add workers and multiproceccing
            self.h = self.model.fit_generator(train_gen, 
                                              validation_data = val_gen, 
                                              epochs = self.epochs)
            del train_gen
            del val_gen
            
        else:
            self.h = self.model.fit(train_in, 
                                    train_out, 
                                    epochs = self.epochs,
                                    validation_split = self.val_split,
                                    batch_size = self.batch_size,
                                    shuffle=True)
        
        loss     = int(round(self.h.history['loss'][-1]))
        val_loss = int(round(self.h.history['val_loss'][-1]))
        
        #Save meta data on a SQL server
        t_stamp = datetime.datetime.now()
        t_stamp = t_stamp.strftime('%d_%b_%y__%H_%M')
        
        self.model.save('../KerasModels/'+t_stamp+f'__{loss}_{val_loss}.hdf5')        
        
# ================================================================================================

    def history_plot(self):
                    
        plt.plot(self.h.history['loss'])
        plt.plot(self.h.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
 
  
    
    