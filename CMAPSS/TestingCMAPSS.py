"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""

import numpy as np
import tensorflow as tf


def read_model(path):
    with open(path + '.json', 'r') as json_file:
        model_json = json_file.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(path + ".h5")

    return model


class Tester:

    @classmethod
    def get_score(cls, models, test_in, true_rul):
        true_rul = true_rul.to_numpy().reshape(-1, 1)

        cls.est_rul = models[0].predict(test_in, batch_size=None).reshape(-1, 1)

        for i in range(1, len(models)):
            cls.est_rul = np.concatenate((cls.est_rul, models[i].predict(test_in, batch_size=None)), axis=1)

        # Calculating S score from the NASA paper, variables can be found there

        pred_err = cls.est_rul - np.repeat(true_rul, len(models), axis=1)

        cls.mse = (pred_err ** 2)
        cls.mse = cls.mse.mean(axis=0)
        cls.pem = pred_err.mean(axis=0)  # Prediction error Mean

        cls.score = np.apply_along_axis(cls._calc_score, 1, pred_err)
        cls.score = np.round(pred_err.sum(axis=0)).astype(int)

        print('Score Summary\n')
        for i in range(len(models)):
            print(
                f'Score is - {cls.score[i]} and MSE is - {cls.mse[i]} and PEM is - {cls.pem[i]} !!! Cry or Celebrate\n')

        if len(models) > 1:
            cm_pred_err = cls.cm_est_rul.mean(axis=1).reshape(-1, 1) - true_rul
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

if __name__ == '__main__':

    from RNNtoFF import RNNtoFF

    number_sequences = 100
    low_range = 80
    high_range = 120

    seq_len = np.random.randint(low_range, high_range, number_sequences)

    sequences = []

    low_val = 0

    high_val = 100

    for i in range(number_sequences):
        sequences.append(np.random.randint(low_val, high_val, (1, seq_len[i], 1)))

    output = np.random.randint(0, 5, number_sequences)

    model_creator = RNNtoFF(1, [5], [3], rnn_type='LSTM')

    model = model_creator.create_trained_model((sequences, output))