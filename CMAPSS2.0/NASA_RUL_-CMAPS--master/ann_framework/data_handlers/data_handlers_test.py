import datetime
import logging
import sys
import numpy as np
import random

sys.path.append('/Users/davidlaredorazo/Documents/University_of_California/Research/Projects')
#sys.path.append('/media/controlslab/DATA/Projects')

from sklearn.preprocessing import OneHotEncoder


from ann_framework.data_handlers.data_handler_CMAPSS import CMAPSSDataHandler
from ann_framework.data_handlers.data_handler_MNIST import MNISTDataHandler
from ann_framework.data_handlers.data_handler_CIFAR10 import CIFAR10DataHandler
from ann_framework.data_handlers.data_handler_Grid import GridDataHandler
from ann_framework.data_handlers.data_handler_NIH import NIHDataHandler
from ann_framework.data_handlers.data_handler_DAMADICS import DamadicsDataHandler


start_date_test = datetime.datetime(2018, 2, 14, 18, 59, 20) # ValveReadingsTest, testing
start_date_training = datetime.datetime(2019, 6, 14, 17, 6, 41) # ValveReadings, trainning
time_delta = datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=1, hours=0, weeks=0)

n = 9800

end_date_training = start_date_training + n*time_delta #get the first n instances
end_date_test = start_date_test + n*time_delta #get the first n instances

features = ['externalControllerOutput', 'undisturbedMediumFlow', 'pressureValveInlet',
            'pressureValveOutlet', 'mediumTemperature', 'rodDisplacement', 'disturbedMediumFlow',
           'selectedFault', 'faultType', 'faultIntensity']

selected_indices = np.array([1,3,4,5,6,7])
selected_features = list(features[i] for i in selected_indices-1)
print(selected_features)

#Does not work for sequence sizes larger than 1 given the way I'm generating the test data.
#Need to properly define what the test data is going to be like.
window_size = 1
window_stride = 1

dHandlder_valve = DamadicsDataHandler(selected_features, window_size, window_stride,
                                      start_date_training=start_date_training, end_date_training=end_date_training,
									  start_date_test=start_date_test, end_date_test=end_date_test,
                                      binary_classes=False, one_hot_encode=False, samples_per_run=10)
dHandlder_valve.connect_to_db('readOnly', '_readOnly2019', '169.236.181.40', 'damadics')

dHandlder_valve.load_data(unroll=True, verbose=1, cross_validation_ratio=0, shuffle_samples=False)
dHandlder_valve.print_data(print_top=True)