# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:01:08 2019

@author: tejas
"""

import Run

prepros_params = {'win_len': 5,
                  'p_order': 3,
                  'std_fac': -0.2,  # Staggered Repetition
                  's_len': 1,  # Unit - Cycle change to percentage of sequence
                  'pca_var': 0.9,
                  'threshold': 1e-5,
                  'denoising': True,
                  'no_splits': 5,
                  'multi_op_normal': True}

model_hparams = {'rnn_type': 'GRU',
                 'rnn_neurons': [60, 60],
                 'ff_neurons': [40, 40]}
train_hparams = {'dropout': 0.3,
                 'rec_dropout': 0.2,
                 'l2_k': 0.06,
                 'l2_b': 0.001,
                 'l2_r': 0.002,
                 'lr': 0.005,
                 'beta': [0.9, 0.999],
                 'epochs': 3,
                 'batch_size': 512,
                 'epsilon': 1e-7,
                 'early_stopping': False,
                 'enable_norm': True,
                 'kcrossval': False}

train_params = {**model_hparams,
                **train_hparams}

cp, rnn_ff, ct = Run.cMAPPS('CMAPSS',
                            prepros_params,
                            train_params,
                            2,
                            tracking=False)
