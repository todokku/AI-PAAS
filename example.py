import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

# %%

[
    ["The", "weather", "will", "be", "nice", "tomorrow"],
    ["How", "are", "you", "doing", "today"],
    ["Hello", "world", "!"]
]

# %%

[
    [83, 91, 1, 645, 1253, 927],
    [73, 8, 3215, 55, 927],
    [71, 1331, 4231]
]

# %%

raw_inputs = [
    [83, 91, 1, 645, 1253, 927],
    [73, 8, 3215, 55, 927],
    [711, 632, 71]
]

# %%

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs,
                                                              padding='post',
                                                              value=1000.0)

padded_inputs = padded_inputs.reshape(-1, 6, 1).astype(float)

# %%

model = tf.keras.Sequential()
model.add(tf.keras.layers.Masking(input_shape=(None, 1), mask_value=1000.0))
model.add(tf.keras.layers.SimpleRNN(1, return_sequences=True))

model.compile(optimizer='Adam', loss='mse', metric='mse')

model.fit(padded_inputs, padded_inputs)

ynew = model.predict(padded_inputs.astype(float))
