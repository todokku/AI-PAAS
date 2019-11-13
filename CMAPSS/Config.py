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
                  'std_fac'   :-0.5,    #Stagered Repetition
                  's_len'     : 2,     #Unit - Cycle change to percentage of sequence
                  'pca_var'   : 0.99,
                  **zero}
    
model_hparams = {'lstm_neurons' : [28,26,24,22,20,18,16,14,12,10],
                 'ff_neurons'   : [8,6]}

train_hparams = {'do_prob'    : 0.4,
                 'l2'         : 0.001,
                 'lr'         : 0.001,
                 'rho'        : 0.8,
                 'val_split'  : 0.30,
                 'epochs'     : 500,
                 'batch_size' : 128,
                 **zero}

train_params = {**sys_params,
                **model_hparams,
                **train_hparams}



#TODO Change to JSON type input later

