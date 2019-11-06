# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Sun Oct 20 13:40:25 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student
    
"""

# ==================================================================================================
# Change Hyperparams here or a module to accept list of hyperparams
# TODO Needs impovement
# ==================================================================================================


sys_params = {'enable_checkp' : False}

#Expose more params to imporve val anf train loss

preprocess_params = {'win_len'   : 21, 
                     'p_order'   : 3, 
                     'threshold' : 1e-5, 
                     's_per'     : 35,    #Stagered Repetition
                     's_len'     : 2,   #Unit - Cycle change to percentage of sequence
                     'pca_var'   : 0.95,
                     'epsilon'   : 1e-8}
    
model_hparams = {'lstm_layer'   : 5, 
                 'lstm_neurons' : 15, 
                 'ff_layer'     : 1, 
                 'ff_neurons'   : 5}

train_hparams = {'val_split'  : 0.35,
                 'epochs'     : 300,
                 'batch_size' : 64}

train_params = {**sys_params,
                **model_hparams,
                **train_hparams}





