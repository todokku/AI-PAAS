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
                  's_len'     : 20,      #Unit - Cycle change to percentage of sequence
                  'pca_var'   : 0.99,
                  'threshold' : 1e-5}
    
model_hparams = {'rnn_type'    : 'simpleRNN',
                 'rnn_neurons' : [28,26,24,22,20,18,16,14,12,10],
                 'ff_neurons'  : [8,6]}

train_hparams = {'dropout'     : 0.4,
                 'rec_dropout' : 0.4,
                 'l2'          : 0.001,
                 'lr'          : 0.01,
                 'beta'        : [0.9,0.999],
                 'val_split'   : 0.30,
                 'epochs'      : 1,
                 'batch_size'  : 128,
                 'epsilon'     : 1e-7}

train_params = {**model_hparams,
                **train_hparams}



#TODO Change to JSON type input later

