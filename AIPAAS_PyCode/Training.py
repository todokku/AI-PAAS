# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Candidate

"""

# =============================================================================
# Choose an appropriate model to create and Train
# =============================================================================

import tensorflow as tf

class LSTM_to_FF:
    
    
    @classmethod
    def config_model(cls, lstm_layer, lstm_neuron, ff_layer, ff_neuron):
        
        cls.lstm_layer = lstm_layer 
        cls.ff_layer   = ff_layer
    
        cls.lstm_neuron = lstm_neuron
        cls.ff_neuron   = ff_neuron
    
    
    
    
    @classmethod
    def create_model(cls, shape):    #shape - (timesteps, features)
        
        cls.shape = shape
        cls.model = tf.keras.models.Sequential()
        
        cls.model.add(tf.keras.layers.Masking(mask_value = 1000.0 ,
                                              input_shape=(None,shape[1])))
        
        for i in range(0, cls.lstm_layer-1):
            
            cls.model.add(tf.keras.layers.LSTM(cls.lstm_neuron, 
                                               return_sequences=True))
    
        cls.model.add(tf.keras.layers.LSTM(cls.lstm_neuron))
        
        for i in range(0, cls.ff_layer):
               
            cls.model.add(tf.keras.layers.Dense(cls.ff_neuron))
            cls.model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
       
        #Final Layer
        cls.model.add(tf.keras.layers.Dense(1))
        
        cls.model.compile(loss='mse',
                          optimizer='adam',
                          metrics=['mape'])
        
#        plot_model(cls.model)
        
        
        
    @classmethod
    def model_train(cls, train_in, train_out):
    
        cls.model.fit(train_in, train_out,
                      batch_size=1, epochs= cls.epochs,
                      validation_split = 0.4,shuffle=True)
        
    
    
    lstm_layer = 1 
    ff_layer   = 1
    
    lstm_neuron = 300
    ff_neuron   = 200
    
    epochs=2
    
    
    
    