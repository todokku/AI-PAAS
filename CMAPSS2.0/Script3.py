import sys

sys.path.append('C:/Users/strix/Documents/Python Scripts/DLRADO/NASA_RUL_-CMAPS--master')

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

from ann_framework.data_handlers.data_handler_CMAPSS import CMAPSSDataHandler

from sklearn.preprocessing import MinMaxScaler

import numpy as np

K.clear_session()  # Clear the previous tensorflow graph

l2_lambda_regularization = 0.20
l1_lambda_regularization = 0.20


def RULmodel_SN_5(input_shape):
    # Create a sequential model
    model = Sequential()

    # Add the layers for the model
    model.add(Dense(20, input_dim=input_shape, activation='relu', kernel_initializer='glorot_normal',
                    kernel_regularizer=regularizers.l2(l2_lambda_regularization),
                    name='fc1'))
    model.add(Dense(20, activation='relu', kernel_initializer='glorot_normal',
                    kernel_regularizer=regularizers.l2(l2_lambda_regularization),
                    name='fc2'))
    model.add(Dense(1, activation='linear', name='out'))

    return model


def get_compiled_model(model):
    # Shared parameters for the models
    optimizer = Adam(learning_rate=0.001, beta_1=0.5)
    lossFunction = "mean_squared_error"
    metrics = ["mse"]
    # Create and compile the models
    model.compile(optimizer=optimizer, loss=lossFunction, metrics=metrics)

    return model


features = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc',
            'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
selected_indices = np.array([2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21])
selected_features = list(features[i] for i in selected_indices - 1)
selected_features.extend(['Op. Settings 1', 'Op. Settings 2', 'Op. Settings 3'])
data_folder = 'C:/Users/strix/Documents/Python Scripts/DLRADO/NASA_RUL_-CMAPS--master/CMAPSSData'
# %%
window_size = 17
window_stride = 1
max_rul = 139

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

dHandler_cmaps = CMAPSSDataHandler(data_folder, 2, selected_features, max_rul,
                                   window_size, window_stride, data_scaler=min_max_scaler)

dHandler_cmaps.load_data(verbose=1, cross_validation_ratio=0)

model = RULmodel_SN_5(dHandler_cmaps.X_train.shape[1])
model = get_compiled_model(model)

print(model.summary())

model.fit(dHandler_cmaps.X_train,
          dHandler_cmaps.y_train,
          epochs=50,
          shuffle=True)

model.evaluate(dHandler_cmaps.X_test,
               dHandler_cmaps.y_test)

pred_values = model.predict(dHandler_cmaps.X_test)

pred_error = pred_values - dHandler_cmaps.y_test


def calc_score(x):
    x[x >= 0] = np.exp(x[x >= 0] / 10 - 1)
    x[x < 0] = np.exp(-(x[x < 0] / 13) - 1)

    return x


score = calc_score(pred_error).sum()
# model.save('C:/Users/strix/Documents/Python Scripts/Models/model4_new.hdf5')
