# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Candidate

"""

# =============================================================================
# Choose an appropriate model to create and Train  # Add saving and plotting capabilities
# =============================================================================

import tensorflow        as tf
import matplotlib.pyplot as plt
import datetime

class LSTM_to_FF:
        
    def __init__(self,
                 features,
                 lstm_layer   = 1, 
                 lstm_neurons = 100, 
                 ff_layer     = 1, 
                 ff_neurons   = 100,
                 epochs       = 10,
                 val_split    = 0.4):
        
        self.lstm_layer = lstm_layer 
        self.ff_layer   = ff_layer
    
        self.lstm_neurons = lstm_neurons
        self.ff_neurons   = ff_neurons
        
        self.features  = features
        self.epochs    = epochs 
        self.val_split = 0.4
        
# =============================================================================
    
    def create_model(self):
        
        #add batch normalization
        
        self.mp = False #No Multiprocessing
        
        self.model = tf.keras.models.Sequential()
        
        self.model.add(tf.keras.layers.Masking(mask_value = 1000.0 ,
                                              input_shape=(None,self.features)))
        
        for i in range(0, self.lstm_layer-1):
            
            self.model.add(tf.keras.layers.LSTM(self.lstm_neurons, 
                                                return_sequences=True))
    
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
        
# =============================================================================
    
    def create_model_multip(self):
        
        #add batch normalization
        
        self.mp = False #No Multiprocessing
        
        self.model = tf.keras.models.Sequential()
        
        self.model.add(tf.keras.layers.Masking(mask_value = 1000.0 ,
                                              input_shape=(None,self.features)))
        
        for i in range(0, self.lstm_layer-1):
            
            self.model.add(tf.keras.layers.LSTM(self.lstm_neurons, 
                                                return_sequences=True))
    
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
        
    def train_model(self, train_in, train_out):
    
        self.h = self.model.fit(train_in, 
                                train_out, 
                                epochs = self.epochs,
                                validation_split = self.val_split, 
                                shuffle=True,
                                use_multiprocessing = self.mp)
        
        loss     = round(self.h.history['loss'][-1])
        val_loss = round(self.h.history['val_loss'][-1])
        
        #Save meta data on a SQL server
        t_stamp = datetime.datetime.now()
        t_stamp = t_stamp.strftime('%d_%b_%y__%H_%M')
        
        self.model.save('../KerasModels/'+t_stamp+f'__{loss}_{val_loss}.hdf5')        
        

    def history_plot(self):
                    
        plt.plot(self.h.history['loss'])
        plt.plot(self.h.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
 
  
    
    