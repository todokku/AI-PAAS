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


sys_params = {'enable_checkp' : False}

zero = {'epsilon' : 1e-5}

prepros_params = {'win_len'   : 21, 
                  'p_order'   : 3, 
                  's_per'     : 35,    #Stagered Repetition
                  's_len'     : 2,     #Unit - Cycle change to percentage of sequence
                  'pca_var'   : 0.95,
                  **zero}
    
model_hparams = {'lstm_layer'   : 5, 
                 'lstm_neurons' : [40,35,30,25,20],
                 'ff_layer'     : 2, 
                 'ff_neurons'   : [15,10]}



train_hparams = {'do_prob'    : 0.4,
                 'l2'         : 0.001,
                 'lr'         : 0.001,
                 'rho'        : 0.8,
                 'val_split'  : 0.35,
                 'epochs'     : 30,
                 'batch_size' : 64,
                 **zero}

train_params = {**sys_params,
                **model_hparams,
                **train_hparams}



#TODO Change to JSON type input later

