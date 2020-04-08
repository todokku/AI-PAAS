import pickle
import numpy as np
from GetCMAPSS import CMAPSS
from Normalising import Normalizer
from DimensionReduction import DimensionReducer
from BatchPrep import PrepFixedInOut
from FaultDetection import FaultDetector

ds_no = 4
cmapss = CMAPSS(ds_no)
cmapss.get_data()
selected_feat = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'W31', 'W32']
train_df = cmapss.Train_input[selected_feat]
e_id_df = cmapss.Train_input['Engine ID']


if ds_no == 2 or ds_no == 4:
    op_cond_df = cmapss.Train_input.iloc[:, 2:5]
    normalizer = Normalizer(6)
else:
    op_cond_df = None
    normalizer = Normalizer(1)

dim_reduce = DimensionReducer(0.99)

train_df = normalizer.normalise(train_df, op_cond_df)

fault_start = FaultDetector(0).get_fault_start(train_df, e_id_df)

# train_2Darray = dim_reduce.reduce_dimensions(train_df.to_numpy())
train_2Darray = train_df.to_numpy()








from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
K.clear_session()  # Clear the previous tensorflow graph
l2_lambda_regularization = 0.20
l1_lambda_regularization = 0.20

def RULmodel(input_shape):
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

window_len = 17
preper = PrepFixedInOut(window_len)
batch_array = preper.prep_train_inputs(train_2Darray, e_id_df.to_numpy())
batch_array = batch_array.reshape(-1, window_len*train_2Darray.shape[1])
out_array = preper.prep_train_outputs(fault_start, e_id_df.to_numpy())

modell = get_compiled_model(RULmodel(window_len*train_2Darray.shape[1]))

modell.fit(batch_array, out_array, epochs=50, shuffle=True)







test_df = cmapss.Test_input[selected_feat]
test_eid_df = cmapss.Test_input['Engine ID']

if ds_no == 2 or ds_no == 4:
    op_cond_df = cmapss.Test_input.iloc[:, 2:5]
else:
    op_cond_df = None

e_id_df = cmapss.Test_input['Engine ID']
test_df = normalizer.normalise(test_df, op_cond_df)
# test_2Darray = dim_reduce.reduce_dimensions(test_df.to_numpy())
test_2Darray = test_df.to_numpy()

batch_test = preper.prep_train_inputs(train_2Darray, e_id_df.to_numpy())
batch_test = batch_array.reshape(-1, window_len*train_2Darray.shape[1])

# modell.predict(test_2Darray)
