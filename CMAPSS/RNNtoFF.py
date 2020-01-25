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


class RNNtoFF:

    def __init__(self, features, rnn_neurons, ff_neurons, rnn_type='simpleRNN', epochs=10, batch_size=32, dropout=0.4,
                 rec_dropout=0.2, l2_k=0.001, l2_b=0., l2_r=0., lRELU_alpha=0.05, lr=0.001, model_dir=None, run_id=None,
                 early_stopping=True, enable_norm=False):

        self.rnn_type = rnn_type
        self.rnn_neurons = rnn_neurons
        self.ff_neurons = ff_neurons

        self.features = features
        self.epochs = epochs
        self.batch_size = batch_size

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

    # ==================================================================================================

    def create_simpleRNN(self, model):

        for i in range(0, len(self.rnn_neurons) - 1):

            model.add(tf.keras.layers.SimpleRNN(self.rnn_neurons[i],
                                                dropout=self.dropout,
                                                recurrent_dropout=self.rec_dropout,
                                                kernel_regularizer=self._l2_k,
                                                bias_regularizer=self._l2_b,
                                                recurrent_regularizer=self._l2_r,
                                                return_sequences=True))
            if self.enable_norm:
                model.add(tf.keras.layers.LayerNormalization())

        model.add(tf.keras.layers.SimpleRNN(self.rnn_neurons[-1],
                                            dropout=self.dropout,
                                            recurrent_dropout=self.rec_dropout,
                                            kernel_regularizer=self._l2_k,
                                            bias_regularizer=self._l2_b,
                                            recurrent_regularizer=self._l2_r))
        if self.enable_norm:
            model.add(tf.keras.layers.LayerNormalization())

    # ==================================================================================================

    def create_LSTM(self, model):

        for i in range(0, len(self.rnn_neurons) - 1):

            model.add(tf.keras.layers.LSTM(self.rnn_neurons[i],
                                           dropout=self.dropout,
                                           recurrent_dropout=self.rec_dropout,
                                           kernel_regularizer=self._l2_k,
                                           bias_regularizer=self._l2_b,
                                           recurrent_regularizer=self._l2_r,
                                           return_sequences=True))
            if self.enable_norm:
                model.add(tf.keras.layers.LayerNormalization())

        model.add(tf.keras.layers.LSTM(self.rnn_neurons[-1],
                                       dropout=self.dropout,
                                       recurrent_dropout=self.rec_dropout,
                                       kernel_regularizer=self._l2_k,
                                       bias_regularizer=self._l2_b,
                                       recurrent_regularizer=self._l2_r))
        if self.enable_norm:
            model.add(tf.keras.layers.LayerNormalization())

    # ==================================================================================================

    def create_GRU(self, model):

        for i in range(0, len(self.rnn_neurons) - 1):

            model.add(tf.keras.layers.GRU(self.rnn_neurons[i],
                                          dropout=self.dropout,
                                          recurrent_dropout=self.rec_dropout,
                                          kernel_regularizer=self._l2_k,
                                          bias_regularizer=self._l2_b,
                                          recurrent_regularizer=self._l2_r,
                                          return_sequences=True))
            if self.enable_norm:
                model.add(tf.keras.layers.LayerNormalization())

        model.add(tf.keras.layers.GRU(self.rnn_neurons[-1],
                                      dropout=self.dropout,
                                      recurrent_dropout=self.rec_dropout,
                                      kernel_regularizer=self._l2_k,
                                      bias_regularizer=self._l2_b,
                                      recurrent_regularizer=self._l2_r))
        if self.enable_norm:
            model.add(tf.keras.layers.LayerNormalization())

    # ==================================================================================================

    def create_model(self):

        model = tf.keras.Sequential(tf.keras.layers.Masking(mask_value=0.,
                                                            input_shape=(None, self.features)))
        self._l2_k = tf.keras.regularizers.l2(l=self.l2_k)
        self._l2_r = tf.keras.regularizers.l2(l=self.l2_r)
        self._l2_b = tf.keras.regularizers.l2(l=self.l2_b)

        if self.rnn_type == 'simpleRNN':
            self.create_simpleRNN(model)
        elif self.rnn_type == 'GRU':
            self.create_GRU(model)
        elif self.rnn_type == 'LSTM':
            self.create_LSTM(model)
        else:
            raise Exception('Invalid RNN Type, choose between simpleRNN or LSTM')

        for i in range(0, len(self.ff_neurons)):

            model.add(tf.keras.layers.Dense(self.ff_neurons[i],
                                            kernel_regularizer=self._l2_k,
                                            bias_regularizer=self._l2_b))

            model.add(tf.keras.layers.LeakyReLU())
            if self.enable_norm: model.add(tf.keras.layers.BatchNormalization())

        # Final Layer
        model.add(tf.keras.layers.Dense(1,
                                        activation='softplus'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr,
                                             beta_1=self.beta[0],
                                             beta_2=self.beta[1],
                                             epsilon=self.epsilon)

        model.compile(loss='mse',
                      optimizer=optimizer)
        self.models.append(model)

        print(self.models[0].summary())

    # ================================================================================================

    def train_model(self,
                    splits_in,
                    splits_out,
                    no_splits):

        # t_stamp = datetime.datetime.now()
        # t_stamp = t_stamp.strftime('%d_%b_%y__%H_%M')

        if self.early_stopping:

            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=50,
                                                          restore_best_weights=True)]
        else:
            callbacks = []

        if self.kcrossval:
            split_index = np.arange(no_splits)
        else:
            split_index = [0]

        si = np.arange(no_splits)

        self.loss = np.array([]).reshape(0, 1)
        self.val_loss = np.array([]).reshape(0, 1)
        self.h = []

        for i in split_index:

            print(f'\nTraining Model{i + 1}\n')

            train_in = np.array([]).reshape(0, splits_in[0].shape[1], splits_in[0].shape[2])
            train_out = np.array([])

            for j in si[:-1]:
                train_in = np.concatenate((train_in, splits_in[j]), axis=0)
                train_out = np.concatenate((train_out, splits_out[j]), axis=0)

            si = np.roll(si, 1)

            val_in = splits_in[len(self.models) - 1 - i]
            val_out = splits_out[len(self.models) - 1 - i]

            h = self.models[i].fit(train_in,
                                   train_out,
                                   validation_data=(val_in, val_out),
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   shuffle=True,
                                   callbacks=callbacks)
            self.h.append(h)

            self.loss = np.append(self.loss, int(round(self.h[i].history['loss'][-1])))
            self.val_loss = np.append(self.val_loss, int(round(self.h[i].history['val_loss'][-1])))

            if self.run_id is not None:
                self.models[i].save_weights(self.model_dir + '/' + self.run_id + f'model{i + 1}.h5')
                model_json = self.models[i].to_json()

                with open(self.model_dir + '/' + self.run_id + f'model{i + 1}.json', "w") as json_file:
                    json_file.write(model_json)

        self.del_loss = np.abs(self.loss - self.val_loss)

        print('\nTraining Summary\n')
        for i in split_index:
            print(f'Model{i + 1}')
            print(f'Loss     = {self.loss[i]}')
            print(f'Val_Loss = {self.val_loss[i]}\n')

    # ================================================================================================

    def history_plot(self):

        plt.plot(self.h.history['loss'])
        plt.plot(self.h.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

# ==================================================================================================
# ==================================================================================================
