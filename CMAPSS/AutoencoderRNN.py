"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""
import tensorflow as tf


class RnnAutoencoder:
    """
    Class used to initialize an autoencoder object

    """

    def __init__(self, features, RNN_type, num_neurons, dropout, rec_dropout, epochs, l2_k=0.001, l2_r=0., l2_b=0.):
        self.features = features
        self.RNN_type = RNN_type
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.epochs = epochs
        self.l2_k = l2_k
        self.l2_r = l2_r
        self.l2_b = l2_b
        self.num_neurons = num_neurons  # list where each number representing number of neurons in each layer

        # Create Model
        if RNN_type == 'LSTM':
            self._LSTM_init()

    def _LSTM_init(self):
        self.auto_encoder = tf.keras.Sequential()

        self.auto_encoder = tf.keras.Sequential(tf.keras.layers.Masking(mask_value=1000.0,
                                                                        input_shape=(None, self.features)))

        self.auto_encoder.add(tf.keras.layers.LSTM(self.num_neurons,
                                                   dropout=self.dropout,
                                                   recurrent_dropout=self.rec_dropout,
                                                   kernel_regularizer=self.l2_k,
                                                   bias_regularizer=self.l2_b,
                                                   recurrent_regularizer=self.l2_r))

        self.auto_encoder.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))

        self.auto_encoder.add(tf.keras.layers.LSTM(self.num_neurons,
                                                   dropout=self.dropout,
                                                   recurrent_dropout=self.rec_dropout,
                                                   kernel_regularizer=self.l2_k,
                                                   bias_regularizer=self.l2_b,
                                                   recurrent_regularizer=self.l2_r))

        self.auto_encoder.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))

        self.auto_encoder.compile('Adam', loss='mse', metrics='mse')

    def train(self, input):

        self.auto_encoder.fit(input, input, epochs=self.epochs)
