import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

#%%

[
  ["The", "weather", "will", "be", "nice", "tomorrow"],
  ["How", "are", "you", "doing", "today"],
  ["Hello", "world", "!"]
]

#%%

[
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 3215, 55, 927],
  [71, 1331, 4231]
]

#%%

raw_inputs = [
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 3215, 55, 927],
  [711, 632, 71]
]

#%%

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs,
                                                              padding='post')

print(padded_inputs)

#%%

model = tf.keras.Sequential()
model.add(tf.keras.layers.Masking(input_shape=(1), mask_value=0))
model.add(tf.keras.layers.SimpleRNN(1, return_sequences=True))

model.compile(optimizer='Adam', loss='mse', metrics='mse')



