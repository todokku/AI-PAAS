# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student

"""

# =========================================================================================
# Choose an appropriate model to create and Train
# =========================================================================================

import tensorflow        as tf
import matplotlib.pyplot as plt
import datetime
#class SimpleRNN_Model()

# ================================================================================================== 
# ==================================================================================================

class RNN_to_FF:
        
    def __init__(self,
                 features,
                 rnn_neurons,
                 ff_neurons,
                 rnn_type     = 'simpleRNN',
                 epochs       = 10,
                 val_split    = 0.2,
                 batch_size   = 32,
                 dropout      = 0.4,   #Dropout Probability
                 rec_dropout  = 0.2,
                 l2           = 0.001,
                 lRELU_alpha  = 0.05,
                 epsilon      = 1e-7,
                 lr           = 0.001,
                 beta         = [0.9,0.999],
                 model_dir     = '../KerasModels/',
                 run_id        = None):
      
    
        self.rnn_type    = rnn_type
        self.rnn_neurons = rnn_neurons
        self.ff_neurons  = ff_neurons
        
        self.features   = features
        self.epochs     = epochs 
        self.val_split  = val_split
        self.batch_size = batch_size
        
        self.dropout     = dropout
        self.rec_dropout = rec_dropout
        self.l2          = l2
        self.lr          = lr
        self.beta        = beta
        
        self.lRELU_alpha = lRELU_alpha
        self.epsilon     = epsilon
        
        self.model_dir  = model_dir 
        self.run_id     = run_id 
        
# ==================================================================================================
        
    def create_simpleRNN(self):
        
        for i in range(0, len(self.rnn_neurons)-1):
            
            l2_reg = tf.keras.regularizers.l2(l=self.l2)
          
            self.model.add(tf.keras.layers.SimpleRNN(self.rnn_neurons[i],
                                                     dropout               = self.dropout,
                                                     recurrent_dropout     = self.rec_dropout,
                                                     kernel_regularizer    = l2_reg,
                                                     bias_regularizer      = l2_reg,
                                                     recurrent_regularizer = l2_reg,
                                                     return_sequences=True))
            self.model.add(tf.keras.layers.LayerNormalization())
    
        self.model.add(tf.keras.layers.SimpleRNN(self.rnn_neurons[i],
                                                 dropout               = self.dropout,
                                                 recurrent_dropout     = self.rec_dropout,
                                                 kernel_regularizer    = l2_reg,
                                                 bias_regularizer      = l2_reg,
                                                 recurrent_regularizer = l2_reg))
        self.model.add(tf.keras.layers.LayerNormalization())
        
# ==================================================================================================        
        
    def create_LSTM(self):
        
        l2_reg = tf.keras.regularizers.l2(l=self.l2)
        
        for i in range(0, len(self.rnn_neurons)-1):
            
            self.model.add(tf.keras.layers.LSTM(self.rnn_neurons[i],
                                                dropout               = self.dropout,
                                                recurrent_dropout     = self.rec_dropout,
                                                kernel_regularizer    = l2_reg,
                                                bias_regularizer      = l2_reg,
                                                recurrent_regularizer = l2_reg,
                                                return_sequences=True))
            self.model.add(tf.keras.layers.LayerNormalization())
    
        self.model.add(tf.keras.layers.LSTM(self.rnn_neurons[i],
                                            dropout               = self.dropout,
                                            recurrent_dropout     = self.rec_dropout,
                                            kernel_regularizer    = l2_reg,
                                            bias_regularizer      = l2_reg,
                                            recurrent_regularizer = l2_reg))
        self.model.add(tf.keras.layers.LayerNormalization())
        
# ==================================================================================================        
    
    def create_model(self):
        
        self.model = tf.keras.models.Sequential(tf.keras.layers.Masking(mask_value = 1000.0,
                                                                        input_shape=(None,self.features)))
   
        
        if self.rnn_type == 'simpleRNN':
            
            self.create_simpleRNN()
            
        elif self.rnn_type == 'LSTM':
            
            self.create_LSTM()
            
        else:
            
            raise Exception('Invalid RNN Type, choose between simpleRNN or LSTM')
            
        l2_reg = tf.keras.regularizers.l2(l=self.l2)
        
        for i in range(0, len(self.ff_neurons)):
            
            l2_reg = tf.keras.regularizers.l2(l=self.l2)
            
            self.model.add(tf.keras.layers.Dense(self.ff_neurons[i],
                                                 kernel_regularizer = l2_reg,
                                                 bias_regularizer   = l2_reg))
            
            self.model.add(tf.keras.layers.LeakyReLU())
            self.model.add(tf.keras.layers.BatchNormalization())
       
        #Final Layer
        self.model.add(tf.keras.layers.Dense(1))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr,
                                             beta_1        = self.beta[0],
                                             beta_2        = self.beta[1],
                                             epsilon       = self.epsilon)
              
        self.model.compile(loss='mse',
                           optimizer = optimizer)
        
        print(self.model.summary())
        
# ================================================================================================
        
    def train_model(self,
                    train_in, 
                    train_out):
        
        t_stamp = datetime.datetime.now()
        t_stamp = t_stamp.strftime('%d_%b_%y__%H_%M')
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor  = 'val_loss', 
                                                      patience = 10,
                                                      restore_best_weights = True)]
  
        self.h = self.model.fit(train_in, 
                                train_out, 
                                epochs           = self.epochs,
                                validation_split = self.val_split,
                                batch_size       = self.batch_size,
                                shuffle          = True,
                                callbacks        = callbacks)
        
        self.loss     = int(round(self.h.history['loss'][-1]))
        self.val_loss = int(round(self.h.history['val_loss'][-1]))
        
        if self.run_id != None:
            
            self.model.save_weights(self.model_dir + '/' + self.run_id + '.h5')
            model_json = self.model.to_json()
            
            with open(self.model_dir + '/' + self.run_id + '.json', "w") as json_file:
                json_file.write(model_json)
                   
#            self.model.save(self.model_dir + '/' + self.run_id + '.hdf5')
        
        
        else:
            
            self.model.save_weights(self.model_dir + t_stamp + f'_{self.loss}_{self.val_loss}' + '.h5')
            model_json = self.model.to_json()
            
            with open(self.model_dir + t_stamp + f'_{self.loss}_{self.val_loss}' + '.json', "w") as json_file:
                json_file.write(model_json)
            
#            self.model.save(self.model_dir + t_stamp + f'_{self.loss}_{self.val_loss}.hdf5')
        
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
