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
    def get_score(cls, model, test_in, true_rul):
        true_rul = true_rul.to_numpy().reshape(-1, 1)

        cls.est_rul = model.predict(test_in, batch_size=None).reshape(-1, 1)
        # Calculating S score from the NASA paper, variables can be found there

        pred_err = cls.est_rul - true_rul

        cls.mse = (pred_err ** 2)
        cls.mse = cls.mse.mean()
        cls.pem = pred_err.mean()  # Prediction error Mean

        cls.score = cls._calc_score(pred_err).sum()

        print(f'Score is - {cls.score} and MSE is - {cls.mse} and MPE is - {cls.pem[i]} !!! Cry or Celebrate\n')

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