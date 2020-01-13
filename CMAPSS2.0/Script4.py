import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

sys.path.append('C:/Users/strix/Documents/Python Scripts/DLRADO/NASA_RUL_-CMAPS--master')
from ann_framework.data_handlers.data_handler_CMAPSS import CMAPSSDataHandler

import matplotlib.pyplot as plt

features = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc',
            'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
selected_indices = np.array([2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21])
selected_features = list(features[i] for i in selected_indices - 1)
data_folder = 'C:/Users/strix/Documents/Python Scripts/DLRADO/NASA_RUL_-CMAPS--master/CMAPSSData'

window_size = 17
window_stride = 1
max_rul = 128

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))


def calc_score(x):
    x[x >= 0] = np.exp(x[x >= 0] / 10 - 1)
    x[x < 0] = np.exp(-(x[x < 0] / 13) - 1)
    return x


def gen_yticks(error_array):
    min_val = np.amin(error_array)
    max_val = np.amax(error_array)
    min_val_yticks_multiplier = math.floor(min_val / 25)
    max_val_yticks_multiplier = math.ceil(max_val / 25)
    min_val_yticks = 25 * min_val_yticks_multiplier
    max_val_yticks = 25 * max_val_yticks_multiplier

    y_ticks_array = np.arange(min_val_yticks, max_val_yticks + 25, 25)

    if y_ticks_array.size > 6:
        min_val_yticks_multiplier = math.floor(min_val / 50)
        max_val_yticks_multiplier = math.ceil(max_val / 50)
        min_val_yticks = 50 * min_val_yticks_multiplier
        max_val_yticks = 50 * max_val_yticks_multiplier

        y_ticks_array = np.arange(min_val_yticks, max_val_yticks + 50, 50)

    return y_ticks_array


def plot_RUL(real_rul, predicted_rul):
    x = np.arange(1, real_rul.shape[0] + 1)
    # e = real_rul - predicted_rul
    index = np.argsort(real_rul, axis=0)
    real_rul = np.sort(real_rul, axis=0)
    predicted_rul = predicted_rul[index, :]
    predicted_rul = predicted_rul.reshape(-1, 1)

    # plt.clf()
    # fig = plt.figure(i)
    # fig.suptitle(f'Real RUL vs Predicted RUL for FD00{i}', fontsize=14)
    # plt.title('Real RUL vs Predicted RUL')
    plt.figure(figsize=(14, 7.5))
    # plt.subplot(2, 1, 1)
    plt.xlabel("Engine Number")
    plt.ylabel("RUL")
    plt.plot(x, predicted_rul, 'bo')
    plt.plot(x, real_rul, 'ro')
    plt.legend(('Predicted RUL', 'Real RUL'), loc='upper right', bbox_to_anchor=(1, 0.5))
    plt.show()

    # y_ticks_array = gen_yticks(e)

    # plt.subplot(2, 1, 2)
    # plt.xlabel("Engine Number")
    # plt.ylabel("$E_{RMS}$")
    # plt.plot(x, e, 'm--')
    # plt.yticks(y_ticks_array)


i = 4

model = tf.keras.models.load_model(f'C:/Users/strix/Documents/Python Scripts/Models/model{i}.hdf5')
dHandler_cmaps = CMAPSSDataHandler(data_folder, i, selected_features, max_rul,
                                   window_size, window_stride, data_scaler=min_max_scaler)

dHandler_cmaps.load_data(verbose=1, cross_validation_ratio=0)

model.evaluate(dHandler_cmaps.X_test,
               dHandler_cmaps.y_test)

pred_values = model.predict(dHandler_cmaps.X_test)

pred_error = pred_values - dHandler_cmaps.y_test
mae = np.abs(pred_error).sum()/pred_error.shape[0]

score = calc_score(pred_error).sum()
plot_RUL(dHandler_cmaps.y_test, pred_values)
