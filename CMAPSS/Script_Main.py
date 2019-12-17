# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:01:08 2019

@author: tejas
"""

import Run


prepros_params = {'win_len'   : 5, 
                  'p_order'   : 3, 
                  'std_fac'   : -0.2,    #Stagered Repetition
                  's_len'     : 2,      #Unit - Cycle change to percentage of sequence
                  'pca_var'   : 0.9,
                  'thresold'  : 1e-5,
                  'denoising' : True}
    
model_hparams = {'rnn_type'    : 'simpleRNN',
                 'rnn_neurons' : [15,15],
                 'ff_neurons'  : [15,15]}

train_hparams = {'dropout'        : 0.4,
                 'rec_dropout'    : 0.2,
                 'l2_k'           : 0.06,
                 'l2_b'           : 0.0,
                 'l2_r'           : 0.002,
                 'lr'             : 0.005,
                 'beta'           : [0.9,0.999],
                 'val_split'      : 0.30,
                 'epochs'         : 5,
                 'batch_size'     : 128,
                 'epsilon'        : 1e-7,
                 'early_stopping' : False,
                 'enable_norm'    : True}

prepros_params = {**prepros_params,
                  'val_split' : train_hparams['val_split']}

train_hparams.pop('val_split')
train_params = {**model_hparams,
                **train_hparams}


Run.cMAPPS('CMAPSS',
           prepros_params,
           train_params,
           1,
           tracking = False)






