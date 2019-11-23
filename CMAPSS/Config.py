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

prepros_params = {'win_len'   : 21, 
                  'p_order'   : 3, 
                  'std_fac'   :-0.5,    #Stagered Repetition
                  's_len'     : 5,      #Unit - Cycle change to percentage of sequence
                  'pca_var'   : 0.90,
                  'thresold'  : 1e-5}
    
model_hparams = {'rnn_type'    : 'simpleRNN',
                 'rnn_neurons' : [2],
                 'ff_neurons'  : []}

train_hparams = {'dropout'        : 0.,
                 'rec_dropout'    : 0.,
                 'l2'             : 0.,
                 'lr'             : 0.002,
                 'beta'           : [0.9,0.999],
                 'val_split'      : 0.30,
                 'epochs'         : 500,
                 'batch_size'     : 128,
                 'epsilon'        : 1e-7,
                 'early_stopping' : False}

train_params = {**model_hparams,
                **train_hparams}



#TODO Change to JSON type input later

