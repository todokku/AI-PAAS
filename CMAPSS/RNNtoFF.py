"""
AI-PAAS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-PAAS Phd Student

"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.compat.v1.disable_eager_execution()


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, in_list, out_array):
        self.in_list = in_list
        self.out_array = out_array

        self.index = np.arange(out_array.shape[0])
        np.random.shuffle(self.index)

    def __len__(self):
        return self.out_array.shape[0]

    def __getitem__(self, ind):
        return self.in_list[self.index[ind]], np.array([self.out_array[ind]])

    def on_epoch_end(self):
        np.random.shuffle(self.index)


class RNNtoFF:

    def __init__(self, features, rnn_neurons=[5, 5], ff_neurons=[5], rnn_type='simpleRNN', epochs=1, lRELU_alpha=0.05,
                 lr=0.001, dropout=0.4, rec_dropout=0.2, l2_k=0.001, l2_b=0., l2_r=0., run_id=None, model_dir=None,
                 early_stopping=False, enable_norm=False):

        self.rnn_type = rnn_type
        self.rnn_neurons = rnn_neurons
        self.ff_neurons = ff_neurons

        self.features = features
        self.epochs = epochs

        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.l2_k = l2_k
        self.l2_r = l2_r
        self.l2_b = l2_b
        self.lr = lr
        self.lRELU_alpha = lRELU_alpha

        self.model_dir = model_dir
        self.run_id = run_id
        self.early_stopping = early_stopping
        self.enable_norm = enable_norm

        self.train_counter = 0

    def _RNN(self, i, return_sequences, input_shape=None):
        if self.rnn_type == 'simpleRNN':
            if input_shape is None:
                return tf.keras.layers.SimpleRNN(self.rnn_neurons[i],
                                                 dropout=self.dropout,
                                                 recurrent_dropout=self.rec_dropout,
                                                 kernel_regularizer=self._l2_k,
                                                 bias_regularizer=self._l2_b,
                                                 recurrent_regularizer=self._l2_r,
                                                 return_sequences=return_sequences)
            else:
                return tf.keras.layers.SimpleRNN(self.rnn_neurons[i],
                                                 input_shape=input_shape,
                                                 dropout=self.dropout,
                                                 recurrent_dropout=self.rec_dropout,
                                                 kernel_regularizer=self._l2_k,
                                                 bias_regularizer=self._l2_b,
                                                 recurrent_regularizer=self._l2_r,
                                                 return_sequences=return_sequences)
        elif self.rnn_type == 'LSTM':
            if input_shape is None:
                return tf.keras.layers.LSTM(self.rnn_neurons[i],
                                            dropout=self.dropout,
                                            recurrent_dropout=self.rec_dropout,
                                            kernel_regularizer=self._l2_k,
                                            bias_regularizer=self._l2_b,
                                            recurrent_regularizer=self._l2_r,
                                            return_sequences=return_sequences)
            else:
                return tf.keras.layers.LSTM(self.rnn_neurons[i],
                                            input_shape=input_shape,
                                            dropout=self.dropout,
                                            recurrent_dropout=self.rec_dropout,
                                            kernel_regularizer=self._l2_k,
                                            bias_regularizer=self._l2_b,
                                            recurrent_regularizer=self._l2_r,
                                            return_sequences=return_sequences)

        elif self.rnn_type == 'GRU':
            if input_shape is None:
                return tf.keras.layers.GRU(self.rnn_neurons[i],
                                           dropout=self.dropout,
                                           recurrent_dropout=self.rec_dropout,
                                           kernel_regularizer=self._l2_k,
                                           bias_regularizer=self._l2_b,
                                           recurrent_regularizer=self._l2_r,
                                           return_sequences=return_sequences)
            else:
                return tf.keras.layers.GRU(self.rnn_neurons[i],
                                           input_shape=input_shape,
                                           dropout=self.dropout,
                                           recurrent_dropout=self.rec_dropout,
                                           kernel_regularizer=self._l2_k,
                                           bias_regularizer=self._l2_b,
                                           recurrent_regularizer=self._l2_r,
                                           return_sequences=return_sequences)
        else:
            raise Exception('Invalid RNN type')

    def _create_RNN(self, model):
        if len(self.rnn_neurons) == 1:
            model.add(self._RNN(0, False, (None, self.features)))
        else:
            model.add(self._RNN(0, True, (None, self.features)))
        if self.enable_norm:
            model.add(tf.keras.layers.LayerNormalization())

        if len(self.rnn_neurons) > 2:
            for i in range(1, len(self.rnn_neurons) - 1):
                model.add(self._RNN(i, True))
                if self.enable_norm:
                    model.add(tf.keras.layers.LayerNormalization())

        if len(self.rnn_neurons) > 1:
            model.add(self._RNN(-1, False))
            if self.enable_norm:
                model.add(tf.keras.layers.LayerNormalization())

        return model

    def _create_model(self):

        model = tf.keras.Sequential()
        self._l2_k = tf.keras.regularizers.l2(l=self.l2_k)
        self._l2_r = tf.keras.regularizers.l2(l=self.l2_r)
        self._l2_b = tf.keras.regularizers.l2(l=self.l2_b)

        model = self._create_RNN(model)

        for i in range(0, len(self.ff_neurons)):

            model.add(tf.keras.layers.Dense(self.ff_neurons[i],
                                            kernel_regularizer=self._l2_k,
                                            bias_regularizer=self._l2_b))

            model.add(tf.keras.layers.LeakyReLU())
            if self.enable_norm:
                model.add(tf.keras.layers.BatchNormalization())

        # Final Layer
        model.add(tf.keras.layers.Dense(1, activation='softplus'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        model.compile(loss='mse',
                      metrics=['mse'],
                      optimizer=optimizer)

        print(model.summary())

        return model

    def create_trained_model(self,
                             train,  # Tuple
                             val=None):  # Tuple

        return self.retrain_model(self._create_model(), train, val)  # retrain just used for simplicity

    def retrain_model(self, model, train, val=None):

        if self.early_stopping:

            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=50,
                                                          restore_best_weights=True)]
        else:
            callbacks = []

        if val is None:
            self.h = model.fit(DataGenerator(train[0], train[1]),
                               epochs=self.epochs,
                               callbacks=callbacks)
        else:
            self.h = model.fit(DataGenerator(train[0], train[1]),
                               validation_data=DataGenerator(val[0], val[1]),
                               epochs=self.epochs,
                               callbacks=callbacks)

        self.loss = int(round(self.h.history['loss'][-1]))
        self.mse = int(round(self.h.history['mse'][-1]))

        if val is not None:
            self.val_loss = int(round(self.h.history['val_loss'][-1]))
            self.del_loss = np.abs(self.loss - self.val_loss)

        if self.run_id is not None:
            self._model_save(model)

        self.train_counter += 1

        return model

    def _model_save(self, model):
        if self.train_counter == 0:
            model.save_weights(self.model_dir + '/' + self.run_id + '.h5')
            model_json = model.to_json()

            with open(self.model_dir + '/' + self.run_id + '.json', "w") as json_file:
                json_file.write(model_json)
        else:
            model.save_weights(self.model_dir + '/' + self.run_id + f'_retrain{self.train_counter}.h5')
            model_json = model.to_json()

            with open(self.model_dir + '/' + self.run_id + f'_retrain{self.train_counter}.json', "w") as json_file:
                json_file.write(model_json)

    def history_plot(self):

        plt.plot(self.h.history['loss'])
        plt.plot(self.h.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()


if __name__ == "__main__":

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

    import tempfile

    tmpdir = tempfile.TemporaryDirectory()
    model_manager = RNNtoFF(1, run_id='1', model_dir=tmpdir.name)

    model = model_manager.create_trained_model((sequences, output))

    model = model_manager.retrain_model(model, (sequences, output))

    tmpdir.cleanup()
