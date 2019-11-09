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

class LSTM_to_FF:
        
    def __init__(self,
                 features,
                 lstm_layer   = 1, 
                 lstm_neurons = 100, 
                 ff_layer     = 1, 
                 ff_neurons   = 100,
                 epochs       = 10,
                 val_split    = 0.2,
                 batch_size   = 32,
                 do_prob      = 0.4,   #Dropout Probability
                 l2           = 0.001,
                 lRELU_alpha  = 0.05,
                 epsilon      = 1e-7,
                 lr           = 0.001,
                 rho          = 0.8,
                 enable_checkp = False,
                 tracking      = False):
# TODO add optimzer based params       
        self.lstm_layer = lstm_layer 
        self.ff_layer   = ff_layer
    
        self.lstm_neurons = lstm_neurons
        self.ff_neurons   = ff_neurons
        
        self.features   = features
        self.epochs     = epochs 
        self.val_split  = val_split
        self.batch_size = batch_size
        
        self.do_prob = do_prob
        self.l2      = l2
        self.lr      = lr
        self.rho     = rho
        
        self.lRELU_alpha = lRELU_alpha
        self.epsilon     = epsilon
        
        self.tracking      = tracking
        self.enable_checkp = enable_checkp
        
# ================================================================================================
    
    def create_model(self):
        
        self.model = tf.keras.models.Sequential()
        
        self.model.add(tf.keras.layers.Masking(mask_value = 1000.0 ,
                                               input_shape=(None,self.features)))
        
        for i in range(0, self.lstm_layer-1):
            #TODO add cuda lstm after testing normal lstm with gpu
            self.model.add(tf.keras.layers.LSTM(self.lstm_neurons[i], return_sequences=True))
            self.model.add(tf.keras.layers.Dropout(self.do_prob))
            self.model.add(tf.keras.layers.ActivityRegularization(l2=self.l2))
            self.model.add(tf.keras.layers.BatchNormalization())
    
        self.model.add(tf.keras.layers.LSTM(self.lstm_neurons[-1]))
        self.model.add(tf.keras.layers.Dropout(self.do_prob))
        self.model.add(tf.keras.layers.ActivityRegularization(l2=self.l2))
        self.model.add(tf.keras.layers.BatchNormalization())
        
        for i in range(0, self.ff_layer):
               
            self.model.add(tf.keras.layers.Dense(self.ff_neurons[i]))
            self.model.add(tf.keras.layers.Dropout(self.do_prob))
            self.model.add(tf.keras.layers.ActivityRegularization(l2=self.l2))
            self.model.add(tf.keras.layers.LeakyReLU(self.lRELU_alpha))
            self.model.add(tf.keras.layers.BatchNormalization())
       
        #Final Layer
        self.model.add(tf.keras.layers.Dense(1))
        
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = self.lr,
                                                rho           = self.rho,
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
        
        #Callbacks
        callbacks = []

#        if self.enable_checkp:
#            
#            path = './KerasModels/'+t_stamp
#            os.mkdir(path)
#            callbacks.append(tf.keras.callbacks.ModelCheckpoint(path + f'__{loss}_{val_loss}__{epoch}.hdf5',
#                                                                'val_loss',
#                                                                save_freq = 10*shape[0])) #change this
#  
        self.h = self.model.fit(train_in, 
                                train_out, 
                                epochs = self.epochs,
                                validation_split = self.val_split,
                                batch_size = self.batch_size,
                                shuffle=True,
                                callbacks = callbacks)
        
        self.loss     = int(round(self.h.history['loss'][-1]))
        self.val_loss = int(round(self.h.history['val_loss'][-1]))
        
        if not(self.tracking):
            self.model.save('../KerasModels/'+ t_stamp + f'_{self.loss}_{self.val_loss}.hdf5')
        
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
        

 
  
    
    