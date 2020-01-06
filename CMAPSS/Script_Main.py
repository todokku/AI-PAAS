# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:01:08 2019

@author: tejas
"""

import Run

prepros_params = {'win_len': 5,
                  'p_order': 3,
                  'std_fac': -0.2,  # Staggered Repetition
                  's_len': 20,  # Unit - Cycle change to percentage of sequence
                  'pca_var': 0.9,
                  'threshold': 1e-5,
                  'denoising': True,
                  'no_splits': 2}

model_hparams = {'rnn_type': 'GRU',
                 'rnn_neurons': [30, 30],
                 'ff_neurons': [20, 20]}

train_hparams = {'dropout': 0.3,
                 'rec_dropout': 0.2,
                 'l2_k': 0.06,
                 'l2_b': 0.001,
                 'l2_r': 0.002,
                 'lr': 0.005,
                 'beta': [0.9, 0.999],
                 'epochs': 1,
                 'batch_size': 32,
                 'epsilon': 1e-7,
                 'early_stopping': False,
                 'enable_norm': True,
                 'kcrossval': True}

train_params = {**model_hparams,
                **train_hparams}

Run.cMAPPS('CMAPSS',
           prepros_params,
           train_params,
           2,
           tracking=False)
