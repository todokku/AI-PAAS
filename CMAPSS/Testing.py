# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student

"""

# =============================================================================
# NOT TO BE USED TO TWEAK THE HYPERPARAMETERS !!!!
# you can but only if you are zen
# =============================================================================


import numpy as np
import tensorflow as tf


def read_model(path):  # path is the common name between the two files

    with open(path + '.json', 'r') as json_file:
        model_json = json_file.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(path + ".h5")

    return model


#    return tf.keras.models.load_model(path)

class cMAPSS:

    @classmethod
    def get_score(cls, models, test_in, true_rul):
        true_rul = true_rul.to_numpy().reshape(-1, 1)

        est_rul = models[0].predict(test_in, batch_size=None).reshape(-1, 1)

        for i in range(len(models) - 1):
            est_rul = np.concatenate((est_rul, models[i].predict(test_in, batch_size=None)), axis=1)

        # Calculating S score from the NASA paper, variables can be found there

        pred_err = est_rul - np.repeat(true_rul, len(models), axis=1)

        cls.mse = (pred_err ** 2)
        cls.mse = cls.mse.mean(axis=0)
        cls.pem = pred_err.mean(axis=0)  # Prediction error Mean

        cls.score = np.apply_along_axis(cls._calc_score, 1, pred_err)
        cls.score = np.round(pred_err.sum(axis=0)).astype(int)

        print('Score Summary\n')
        for i in range(len(models)):
            print(f'Score is - {cls.score[i]} and MSE is - {cls.mse[i]} and PEM is - {cls.pem[i]} !!! Cry or Celebrate\n')

        if len(models) > 1:
            cm_pred_err = est_rul.mean(axis=1).T - true_rul
            cls.cm_mse = (cm_pred_err ** 2)
            cls.cm_mse = cls.cm_mse.mean()
            cls.cm_pem = cm_pred_err.mean()
            cls.cm_score = cls._calc_score(cm_pred_err)
            cls.cm_score = np.round(cm_pred_err.sum()).astype(int)
            print(f'Combined Score is - {cls.cm_score} and MSE is - {cls.cm_mse} and PEM is - {cls.cm_pem} !!! Cry or '
                  'Celebrate\n')

    @classmethod
    def _calc_score(cls, x):

        x[x >= 0] = np.exp(x[x >= 0] / 10 - 1)
        x[x < 0] = np.exp(-(x[x < 0] / 13) - 1)

        return x

    def __init__(self):
        raise Exception('Cannot Create new Object')
