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

    def __init__(self):
        pass

    def get_score(self, model, test_list, true_rul_array):
        true_rul_array = true_rul_array.reshape(-1, 1)
        self.est_rul = model.predict(test_list, batch_size=None).reshape(-1, 1)
        # Calculating S score from the NASA paper, variables can be found there
        pred_err = self.est_rul - true_rul_array

        self.mse = (pred_err ** 2)
        self.mse = self.mse.mean()
        self.mpe = pred_err.mean()  # Mean Prediction Error

        score = self._calc_score(pred_err).sum()
        print(f'\nScore is - {score} and MSE is - {self.mse} and MPE is - {self.mpe} !!! Cry or Celebrate\n')

        return score

    def _calc_score(self, x):
        x[x >= 0] = np.exp(x[x >= 0] / 10 - 1)
        x[x < 0] = np.exp(-(x[x < 0] / 13) - 1)

        return x

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

    tester = Tester()
    score = tester.get_score(model, sequences, output)
