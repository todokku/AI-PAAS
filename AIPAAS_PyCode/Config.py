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
# ==================================================================================================


sys_params = {'use_gen'       : True,
              'enable_checkp' : False}

#Expose more params to imporve score

preprocess_params = {'win_len'   : 21, 
                     'p_order'   : 3, 
                     'threshold' : 1e-5, 
                     's_rep'     : 3,    #Stagered Repetition
                     's_len'     : 40,   #Unit - Cycle 
                     'pca_var'   : 0.97,
                     'epsilon'   : 1e-8}

model_hparams = {'lstm_layer'   : 3, 
                 'lstm_neurons' : 300, 
                 'ff_layer'     : 2, 
                 'ff_neurons'   : 150}

train_hparams = {'val_split'  : 0.4,
                 'epochs'     : 40,
                 'batch_size' : 64}












