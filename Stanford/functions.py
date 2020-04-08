import numpy as np
import pandas as pd
import sklearn.linear_model as skll
import matplotlib.pyplot as plt

import scipy.interpolate as sci_int


# Data Group
# train
# test1
# test2

def input_data(d_g):
    c_lives = pd.read_csv(f'./data/cycle_lives/{d_g}_cycle_lives.csv', names=['Cycle_Lives']).to_numpy()
    n_cells = c_lives.size

    c_v_arrays = np.zeros((n_cells, 99, 1000))
    voltage_names = []

    for i in range(99):
        voltage_names.append(f'C{i + 1}')

    for i in range(n_cells):
        df = pd.read_csv(f'./data/{d_g}/cell{i + 1}.csv', names=voltage_names)
        c_v_arrays[i, :, :] = df.to_numpy().T

    return c_v_arrays, c_lives


def sci_univar(y):
    return sci_int.UnivariateSpline(np.arange(1000), y, k=3, s=1000)(np.arange(1000))


def delQ(c_v_arrays, max_, min_):
    delQ_ij = np.zeros((c_v_arrays.shape[0], 1000))

    for i in range(c_v_arrays.shape[0]):
        delQ_ij[i, :] = c_v_arrays[i, max_, :] - c_v_arrays[i, min_, :]

    return delQ_ij


c_v_array, cc_lives = input_data('train')
c_v_array = np.apply_along_axis(sci_univar, 1, c_v_array.reshape(-1, 1000)).reshape(-1, 99, 1000)
delQQ = delQ(c_v_array, 98, 8)

varQ = np.log10(delQQ.var(axis=1))
minQ = np.log10(np.abs(delQQ.min(axis=1)))

train_X = np.concatenate((varQ.reshape(-1, 1), minQ.reshape(-1, 1)), axis=1)
train_X = varQ.reshape(-1, 1)
model = skll.ElasticNetCV(cv=4).fit(train_X, cc_lives.reshape(-1))

score = model.score(train_X, cc_lives)

c_v_array, cc_lives = input_data('test1')
c_v_array = np.apply_along_axis(sci_univar, 1, c_v_array.reshape(-1, 1000)).reshape(-1, 99, 1000)
delQQ = delQ(c_v_array, 98, 8)

varQ = np.log10(delQQ.var(axis=1))
minQ = np.log10(np.abs(delQQ.min(axis=1)))

test_X = np.concatenate((varQ.reshape(-1, 1), minQ.reshape(-1, 1)), axis=1)
test_X = varQ.reshape(-1, 1)

score_test = model.score(test_X, cc_lives)
pred = model.predict(test_X)

plt.scatter(cc_lives, model.predict(test_X))
plt.plot(np.arange(0, 2000), np.arange(0, 2000))
plt.xlabel('Observed Cycle Life')
plt.ylabel('Predicted Cycle Life')
plt.show()
