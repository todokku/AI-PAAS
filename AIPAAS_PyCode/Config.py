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

#Expose more params to imporve score during 

preprocess_params = {'win_len'   : 21, 
                     'p_order'   : 3, 
                     'threshold' : 1e-5, 
                     's_rep'     : 6,    #Stagered Repetition
                     's_len'     : 20,   #Unit - Cycle 
                     'pca_var'   : 0.97,
                     'epsilon'   : 1e-8}
    
model_hparams = {'lstm_layer'   : 4, 
                 'lstm_neurons' : 500, 
                 'ff_layer'     : 2, 
                 'ff_neurons'   : 400}

train_hparams = {'val_split'  : 0.4,
                 'epochs'     : 60,
                 'batch_size' : 32}












