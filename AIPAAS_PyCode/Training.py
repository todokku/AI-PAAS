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

from   keras.models      import Sequential

from   keras.layers      import Dense
from   keras.layers      import LSTM
from   keras.layers      import Masking
from   keras.layers      import LeakyReLU

class LSTM_to_FF:
    
    @classmethod
    def create_model(cls, shape):
        
        cls.model = Sequential()
        
        model.add(Masking(mask_value = 1000.0 ,input_shape=(cp.max_cycles,cp.train_in.shape[2])))
        model.add(LSTM(140))
    
    
        model.add(Dense(70))
        model.add(LeakyReLU(alpha=0.05))
        #Final Layer
    
        model.add(Dense(1))
    
    
        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mape'])
    
    @classmethod
    def model_train(cls):
    
        model.fit(train_in, train_out,
                  batch_size=1, epochs=10,
                  validation_split = 0.4,shuffle=True)
        
    
    
    
    no_of_lstm = 1
    no_of_ff   = 1 # Not including output
    
    