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

prepros_params = {'win_len'   : 11, 
                  'p_order'   : 3, 
                  'std_fac'   : -1,    #Stagered Repetition
                  's_len'     : 4,      #Unit - Cycle change to percentage of sequence
                  'pca_var'   : 0.97,
                  'thresold'  : 1e-5}
    
model_hparams = {'rnn_type'    : 'simpleRNN',
                 'rnn_neurons' : [1],
                 'ff_neurons'  : []}

train_hparams = {'dropout'        : 0.,
                 'rec_dropout'    : 0.,
                 'l2'             : 0.,
                 'lr'             : 0.002,
                 'beta'           : [0.9,0.999],
                 'val_split'      : 0.20,
                 'epochs'         : 20,
                 'batch_size'     : 128,
                 'epsilon'        : 1e-7,
                 'early_stopping' : False}




prepros_params = {**prepros_params,
                  'val_split' : train_hparams['val_split']}

train_hparams.pop('val_split')
train_params = {**model_hparams,
                **train_hparams}



#TODO Change to JSON type input later

