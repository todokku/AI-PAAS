# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student

"""

# =============================================================================
# NOT TO BE USED TO TWEAK THE HYPERPARAMETERS !!!!
# you can but only if you are zen
# =============================================================================


import numpy      as np
import tensorflow as tf



def read_model(path): #path is the common name between the two files

    with open(path + '.json', 'r') as json_file:
        model_json = json_file.read()
        
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(path + ".h5")
    
    return model
    
    
#    return tf.keras.models.load_model(path)

class cMAPSS:
    
    @classmethod
    def get_score(cls, model, test_in, true_rul):
        
        cls.true_rul = true_rul.to_numpy()
        cls.est_rul  = model.predict(test_in, batch_size=None)  
                
        #Calculating S score from the NASA paper, variables can be found there
        
        d = cls.est_rul - cls.true_rul
        
        cls.mse = (d**2)
        cls.mse = cls.mse.mean()
        
        cls.pem = d.mean()  #Prediction error Mean
        
        d[d>=0]  = np.exp(d[d>=0]/10 - 1) 
        d[d<0]   = np.exp(-(d[d<0]/13) - 1) 

        cls.s    = int(np.round(d.sum()))
        
        print(f'The score is - {cls.s} and mse is - {cls.mse} and pme is - {cls.pem} !!! Cry or Celebrate')
        
    def __init__(self):
        
        raise Exception('Cannot Create new Object')
        
        
        
        
        
        
        
        
        
        