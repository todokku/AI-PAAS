# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 02:58:43 2019

@author: tejas
"""

import pandas as pd
#import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split

#%%

# read in the csv data into a pandas data frame and set the date as the index
df = pd.read_csv('./end-part2_df.csv').set_index('date')

# execute the describe() function and transpose the output so that it doesn't overflow the width of the screen
df_describe = df.describe().T

#%%

# execute the info() function
df.info()

#%%

# First drop the maxtempm and mintempm from the dataframe
df = df.drop(['mintempm', 'maxtempm'], axis=1)

# X will be a pandas dataframe of all columns except meantempm
X = df[[col for col in df.columns if col != 'meantempm']]

# y will be a pandas series of the meantempm
y = df['meantempm']

#%%

# split data into training set and a temporary set using sklearn.model_selection.traing_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.28, random_state=23)

# take the remaining 20% of data in X_tmp, y_tmp and split them evenly
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.75, random_state=23)

X_train.shape, X_test.shape, X_val.shape
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))

del X, y, X_tmp, y_tmp

#%%

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test  = X_test.to_numpy()
y_test  = y_test.to_numpy()
X_val   = X_val.to_numpy()
y_val   = y_val.to_numpy()

model = tf.keras.models.Sequential()
        
model.add(tf.keras.layers.Dense(50, input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.LeakyReLU(0.05))

model.add(tf.keras.layers.Dense(30))
model.add(tf.keras.layers.LeakyReLU(0.05))

model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.LeakyReLU(0.05))

model.add(tf.keras.layers.Dense(1))

model.compile(loss='mse',
              optimizer = 'adam')

print(model.summary())

#%%

h = model.fit(X_train, 
              y_train,
              validation_data=(X_val, y_val),
              epochs = 400,
              batch_size = 128)

#%%

import matplotlib.pyplot as plt

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#%%

predictions = model.predict(X_test, batch_size=None)  

print("The Explained Variance: %.2f" % explained_variance_score(
                                            y_test, predictions))  
print("The Mean Absolute Error: %.2f degrees Celcius" % mean_absolute_error(
                                            y_test, predictions))  
print("The Median Absolute Error: %.2f degrees Celcius" % median_absolute_error(
                                            y_test, predictions))

#%%






#plt.plot()



















