import numpy as np
import random
import pandas as pd
import sqlalchemy
import math
from sqlalchemy.orm import sessionmaker

from datetime import datetime

from ann_framework.data_handlers.damadicsDBMapping import *
from .sequenced_data_handler import SequenceDataHandler

from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

# IP Address: 169.236.181.40
# User: readOnly
# Password: _readOnly2019
# Database: damadics

class DamadicsDataHandler(SequenceDataHandler):

	'''
	TODO: column information here
	'''

	#Method definition

	def __init__(self, selected_features, sequence_length = 1, sequence_stride = 1, data_scaler = None, **kwargs):

		#Public properties
		self._data_scaler = data_scaler
		self._selected_features = selected_features

		#kwargs

		if 'start_date_training' in kwargs:
			self._start_date_training = kwargs['start_date_training']
		else:
			self._start_date_training = None

		if 'end_date_training' in kwargs:
			self._end_date_training = kwargs['end_date_training']
		else:
			self._end_date_training = None

		if 'start_date_test' in kwargs:
			self._start_date_test = kwargs['start_date_test']
		else:
			self._start_date_test = None

		if 'end_date_test' in kwargs:
			self._end_date_test = kwargs['end_date_test']
		else:
			self._end_date_test = None

		if 'one_hot_encode' in kwargs:
			self._one_hot_encode = kwargs['one_hot_encode']
		else:
			self._one_hot_encode = False

		if 'binary_classes' in kwargs:
			self._binary_classes = kwargs['binary_classes']
		else:
			self._binary_classes = False

		if 'samples_per_run' in kwargs:
			self._samples_per_run = kwargs['samples_per_run']
		else:
			self._samples_per_run = 10

		# Database connection
		self._load_from_db_training = True
		self._load_from_db_test = True

		self._column_names = {0: 'timestamp', 1: 'externalControllerOutput', 2: 'undisturbedMediumFlow', 3: 'pressureValveInlet', 4:'pressureValveOutlet',
							  5: 'mediumTemperature', 6: 'rodDisplacement', 7: 'disturbedMediumFlow', 8: 'selectedFault', 9: 'faultType', 10: 'faultIntensity'}

		#feature_size = 6

		# Entire Dataset
		self._df_training = None
		self._df_X_training = None
		self._df_y_training = None
		self._num_samples_training = None
		self._sample_indices_training = None

		self._df_test = None
		self._df_X_test = None
		self._df_y_test = None
		self._num_samples_test = None
		self._sample_indices_test = None

		# Splitting. This is what is used to train

		self._df_test = None

		#create one time session
		self._sqlsession = None

		#super init
		super().__init__(sequence_length=sequence_length, sequence_stride=sequence_stride, feature_size=len(selected_features), data_scaler=data_scaler)


	def connect_to_db(self, username, pasw, host, dbname):

		self.dbname = dbname
		databaseString = "mysql+mysqldb://"+username+":"+pasw+"@"+host+"/"+dbname

		self._sqlsession = None
		try:
			sqlengine = sqlalchemy.create_engine(databaseString)
			SQLSession = sessionmaker(bind=sqlengine)
			self._sqlsession = SQLSession()
			print("Connection to " + databaseString + " successfull")
		except Exception as e:
			print("e:", e)
			print("Error in connection to the database")


	def extract_data_from_db(self, training_data=False):

		y_col_name = 'selectedFault'

		computation_start_time = datetime.now()

		if training_data == True:

			print("Reading data from ValveReading")
			query = self._sqlsession.query(ValveReading).filter(ValveReading._timestamp.between(self._start_date_training, self._end_date_training))

			print(self._start_date_training)
			print(self._end_date_training)

			self._df_training = pd.read_sql(query.statement, self._sqlsession.bind)

			if self._df_training.shape[0] == 0:
				print("No data found for ValveReading dates between {} and {}. Aborting.\n".format(self._start_date_training, self._end_date_training))
				return -1

			self._df_X_training = self._df_training.loc[:, self._selected_features].values
			self._df_y_training = self._df_training.loc[:, [y_col_name]].values
			self._df_y_training = self._df_y_training.reshape(-1, 1)

		else:
			#if (self._start_date_test is not None) and (self._start_date_test is not None):
			print("Reading data from ValveReadingTest")
			query = self._sqlsession.query(ValveReadingTest).filter(ValveReadingTest._timestamp.between(self._start_date_test, self._end_date_test))

			self._df_test = pd.read_sql(query.statement, self._sqlsession.bind)

			if self._df_test.shape[0] == 0:
				print("No data found for ValveReadingTest dates between {} and {}. Aborting.\n".format(self._start_date_test, self._end_date_test))
				return -1

			self._df_X_test = self._df_test.loc[:, self._selected_features].values
			self._df_y_test = self._df_test.loc[:, [y_col_name]].values
			self._df_y_test = self._df_y_test.reshape(-1, 1)

		print("Extracting data from database runtime:", datetime.now() - computation_start_time)

		return 1

	def retrieve_samples_isolated(self, training_data=False):
		"""
		Some assumptions are made for finding the samples. A sample is defined as a cycle current state - change of state.
		That is, the valve starts in some condition, after a while its condition changes and then a new cycle is created.

		1.) The valve starting state can be at any state.
		"""

		if training_data == True:
			labels = self._df_y_training
		else:
			labels = self._df_y_test

		start_indices, fault_indices = list(), list()
		discarded_top_index = 0
		discarded_bottom_index = 0
		i = 0

		# Iterate over the list since no more efficient way was found

		curr_state = labels[i,0]
		prev_state = curr_state
		start_indices.append(i)

		for i in range(i+1, len(labels)):

			curr_state = labels[i,0]

			if prev_state != curr_state:
				start_indices.append(i)

			prev_state = curr_state

		return start_indices, fault_indices, discarded_top_index, discarded_bottom_index


	def retrieve_samples(self, training_data=False):
		"""
		Some assumptions are made for finding the samples. A sample is defined as a cycle normal-failure-stop.
		That is, the valve starts in normal conditions, after a while it fails and remains at fault for an indefinite amount of time.

		1.) The valve status always starts as Normal (faultType = 20).
		2.) Only complete samples are considered, at the end the code reports if unusued chunks of data remained.
		3.) The last data chunk ends when the system is restarted (back to non-faulty state)
		"""

		if training_data == True:
			labels = self._df_y_training
		else:
			labels = self._df_y_test

		start_indices, fault_indices = list(), list()
		discarded_top_index = 0
		discarded_bottom_index = 0
		prev_state = 20
		curr_state = 0
		i = 0
		num_instances = len(labels)

		s_time = datetime.now()

		# Iterate over the list since no more efficient way was found

		#Find the first non-faulty state of the valve and start from there
		curr_state = labels[i]

		while(curr_state != 20 and i < num_instances):
			i = i+1
			curr_state = labels[i]

		discarded_top_index = i-1

		start_indices.append(i)
		prev_state = curr_state

		for i in range(i+1, len(labels)):


			if prev_state != 20 and labels[i] == 20:
				start_indices.append(i)
			elif prev_state == 20 and labels[i] != 20:
				fault_indices.append(i)
			else:
				pass

			prev_state = labels[i]

		#Discard last chunk of data in case state is not fault (since we assume its incomplete)
		if prev_state == 20:
			discarded_bottom_index = start_indices[-1]
			#start_indices.pop()

		return start_indices, fault_indices, discarded_top_index, discarded_bottom_index


	def create_cycles(self, training_data=True, isolated=True):
		"""From the data extracted from the DB, create the valve cycles"""

		# Obtain the indices for each cycle depending on wether the cycles are isolated or not.
		if isolated == False:
			start_indices, fault_indices, discarded_top_index, discarded_bottom_index = self.retrieve_samples(training_data=True)

			indices_shifted = start_indices[1:]
			fault_indices.append(0)
			indices_shifted.append(0)

			indices = [index for index in zip(start_indices, indices_shifted, fault_indices)]

		else:
			start_indices, fault_indices, _, _ = self.retrieve_samples_isolated(training_data=training_data)

			indices_shifted = start_indices[1:]
			indices_shifted.append(0)

			indices = [index for index in zip(start_indices, indices_shifted)]

		# Drop last sample as there is no guarantee that the last one is complete. (need to fix this in case it is complete)
		indices.pop()

		if training_data == True:

			self._sample_indices_training = indices
			self._num_samples_training = len(self._sample_indices_training)

		else:

			self._sample_indices_test = indices
			self._num_samples_test = len(self._sample_indices_test)


	def load_data_training(self, start_date, end_date, categories, cross_validation_ratio = 0, verbose = 0):

		if self._start_date_training != start_date or self._end_date_training != end_date:
			print("Reload from DB")
			self._start_date_training = start_date
			self._end_date_training = end_date
			self._load_from_db_training = True

		if verbose == 1:
			print("Loading training data for DAMADICS with window_size of {}, stride of {}. Cros-Validation ratio {}".format(self._sequence_length, self._sequence_stride, cross_validation_ratio))

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		if self._load_from_db_training == True:
			print("Loading data from database")

			# Attempt to extract the data from the DB
			if self.extract_data_from_db(training_data=True) == -1:
				return None, None

			self.create_cycles(training_data=True, isolated=True)

			# Classes are true or false only
			if self._binary_classes == True:
				self._df_y_training = np.array([-1 if int(y_training[0]) == 20 else 1 for y_training in self._df_y_training])
				self._df_y_training = self._df_y_training.reshape(-1, 1)
				categories = np.array([-1, 1])

			if self._one_hot_encode == True:
				encoder = OneHotEncoder(categories=[categories])
				self._df_y_training = encoder.fit_transform(self._df_y_training).toarray()

		else:
			print("Loading data from memory")

		# Split up the data into its different samples
		# Modify properties in the parent class, and let the parent class finish the data processing
		train_indices = self._sample_indices_training
		cv_indices = []

		if cross_validation_ratio != 0:
			train_indices, cv_indices = self.split_samples(train_indices, cross_validation_ratio, self._num_samples_training)

		self._load_from_db_training = False  # As long as the dataframe doesnt change, there is no need to reload from file

		return train_indices, cv_indices


	def load_data_test(self, start_date, end_date, categories, verbose = 0):

		if self._start_date_test != start_date or self._end_date_test != end_date:
			print("Reload from DB")
			self._start_date_test = start_date
			self._end_date_test = end_date
			self._load_from_db_test = True

		if verbose == 1:
			print("Loading test data for DAMADICS with window_size of {}, stride of {}".format(self._sequence_length, self._sequence_stride))


		if self._load_from_db_test == True:
			print("Loading data from database")

			# These variables are where the entire data is saved at
			if self.extract_data_from_db(training_data=False) == -1:
				return None

			self.create_cycles(training_data=False, isolated=True)

			# Classes are true or false only
			if self._binary_classes == True:
				self._df_y_test = np.array([-1 if int(y_test[0]) == 20 else 1 for y_test in self._df_y_test])
				self._df_y_test = self._df_y_test.reshape(-1, 1)
				categories = np.array([-1, 1])

			if self._one_hot_encode == True:
				encoder = OneHotEncoder(categories=[categories])
				self._df_y_test = encoder.fit_transform(self._df_y_test).toarray()

		else:
			print("Loading data from memory")

		# Split up the data into its different samples
		# Modify properties in the parent class, and let the parent class finish the data processing
		test_indices = self._sample_indices_test

		self._load_from_db_test = False  # As long as the dataframe doesnt change, there is no need to reload from file

		return test_indices


	# Public
	def load_data(self, unroll = True, cross_validation_ratio = 0, verbose = 0, **kwargs):
		"""Load the data using the specified parameters"""

		categories = np.arange(1, 21)

		if 'start_date_training' in kwargs:
			start_date_training = kwargs['start_date_training']
		else:
			start_date_training = self._start_date_training

		if 'end_date_training' in kwargs:
			end_date_training = kwargs['end_date_training']
		else:
			end_date_training = self._end_date_training

		if 'start_date_test' in kwargs:
			start_date_test = kwargs['start_date_test']
		else:
			start_date_test = self._start_date_test

		if 'end_date_test' in kwargs:
			end_date_test = kwargs['end_date_test']
		else:
			end_date_test = self._end_date_test

		if 'shuffle_samples' in kwargs:
			shuffle_samples = kwargs['shuffle_samples']
		else:
			shuffle_samples = True


		train_indices, cv_indices = self.load_data_training(start_date_training, end_date_training, categories, cross_validation_ratio, verbose)
		test_indices = self.load_data_test(start_date_test, end_date_test, categories, verbose)

		if (train_indices is None) or (test_indices is None):
			return
		else:
			self._X_train_list, self._y_train_list, self._X_crossVal_list, self._y_crossVal_list, self._X_test_list, self._y_test_list = \
				self.generate_lists(train_indices, cv_indices, test_indices, self._samples_per_run)

			self.generate_train_data(unroll)

			if cross_validation_ratio != 0:
				self.generate_crossValidation_data(unroll)

			self.generate_test_data(unroll)

			#shuffle the data
			if shuffle_samples:

				self._X_train, self._y_train = shuffle(self._X_train, self._y_train)

				if cross_validation_ratio != 0:
					self._X_crossVal, self._y_crossVal = shuffle(self._X_crossVal, self._y_crossVal)

				self._X_test, self._y_test = shuffle(self._X_test, self._y_test)


	# Private
	def split_samples(self, indices, split_ratio, num_samples):
		''' From the dataframes generate the feature arrays and their labels'''

		startTime = datetime.now()

		shuffled_samples = list(range(0, num_samples))
		random.shuffle(shuffled_samples)

		X_train_list, y_train_list = list(), list()
		X_crossVal_list, y_crossVal_list = list(), list()
		X_test_list, y_test_list = list(), list()

		if (split_ratio < 0 or split_ratio > 1):
			print("Error, split ratio must be between 0 and 1")
			return

		num_split_test = math.ceil(split_ratio*num_samples)
		num_split_train = num_samples - num_split_test

		if num_split_train == 0 or num_split_test == 0:
			print("Error: one of the two splits is left with 0 samples")
			return

		indices_train = shuffled_samples[:num_split_train]
		indices_test = shuffled_samples[num_split_train:]

		samples_train = [indices[i] for i in indices_train]
		samples_test = [indices[i] for i in indices_test]

		print("Data Splitting:",datetime.now() - startTime)

		return samples_train, samples_test


	def generate_lists(self, train_indices, cv_indices, test_indices, num_samples_per_run):
		"""Given the indices generate the lists from the dataframe"""

		"""TODO: How to define the status of run when it could be faulty or non faulty? Take the mayority?, take piecewise linear? (treshold), take last?"""

		rnd_index = 0
		count_samples = 0
		count_attempts = 0
		max_attempts = 5

		indices_set = set()

		train_list_X, train_list_y = list(), list()
		cv_list_X, cv_list_y = list(), list()
		test_list_X, test_list_y = list(), list()

		sample_x = None
		sample_y = None

		#print(train_indices)
		#print(cv_indices)
		#print(test_indices)

		for indices in train_indices:

			sample_x = self._df_X_training[indices[0]:indices[1], :]
			sample_y = self._df_y_training[indices[0]:indices[1], :] # Since this is training data, status of the valve is left untouched.

			train_list_X.append(sample_x)
			train_list_y.append(sample_y)
		train_numbers = len(train_list_X)

		#print("number of training data")
		#print(train_numbers)

		for indices in cv_indices:

			count_attempts = 0
			count_samples = 0
			indices_set = set()

			#print("CV indices")

			#Avoid repeated samples
			while (count_samples < num_samples_per_run) and (count_attempts < max_attempts):

				# Attempt to obtain a sample
				start_index, stop_index = self.get_test_sample_indices(indices)
				#print((start_index, stop_index))

				if (stop_index != 0) and ((start_index, stop_index) not in indices_set):

					sample_x = self._df_X_training[start_index:stop_index, :]
					sample_y = self._df_y_training[stop_index-1:stop_index, :]   #Status of the valve for the run is the last fault indication that appeared

					indices_set.add((start_index, stop_index))
					cv_list_X.append(sample_x)
					cv_list_y.append(sample_y)

					count_samples = count_samples + 1
				else:
					#print("Repeated index")
					count_attempts = count_attempts + 1

		#Test data is an instance of size sequence_size for each sample
		for indices in test_indices:

			count_attempts = 0
			count_samples = 0
			indices_set = set()

			#print("Test indices")

			# Avoid repeated samples
			while (count_samples < num_samples_per_run) and (count_attempts < max_attempts):

				# Attempt to obtain a sample
				start_index, stop_index = self.get_test_sample_indices(indices)
				#print((start_index, stop_index))

				if (stop_index != 0) and ((start_index, stop_index) not in indices_set):

					sample_x = self._df_X_test[start_index:stop_index, :]
					sample_y = self._df_y_test[stop_index-1:stop_index, :] #Status of the valve for the run is the last fault indication that appeared

					indices_set.add((start_index, stop_index))
					test_list_X.append(sample_x)
					test_list_y.append(sample_y)

					count_samples = count_samples + 1
				else:
					#print("Repeated index")
					count_attempts = count_attempts + 1


		return train_list_X, train_list_y, cv_list_X, cv_list_y, test_list_X, test_list_y


	def get_test_sample_indices(self, sample_indices, isolated=True):
		"""Attempt to get the indices for a test sample"""

		start_index = 0
		stop_index = 0

		if sample_indices[1] - (sample_indices[0] + self.sequence_length) <= 0:
			print("Cylce for {} is less than sequence lenght".format(sample_indices))
			return 0, 0

		rnd_index = random.randint(sample_indices[0] + self.sequence_length, sample_indices[1])

		if isolated == False:

			#If the rand index is before the fault point. Sample is taken from healty states.
			if rnd_index < sample_indices[2]:
				rnd_index2 = rnd_index - self.sequence_length

				if rnd_index2 >= sample_indices[0]:
					start_index = rnd_index2
					stop_index = rnd_index
			#Take sample from faulty states.
			elif rnd_index > sample_indices[2]:
				rnd_index2 = rnd_index + self.sequence_length

				if rnd_index2 <= sample_indices[1]:
					start_index = rnd_index
					stop_index = rnd_index2
			else:
				start_index = 0
				stop_index = 0

		else:
			#Find a suitable stop index
			start_index = rnd_index
			stop_index = rnd_index + self.sequence_length

			if stop_index > sample_indices[1]:
				start_index = rnd_index - self.sequence_length
				stop_index = rnd_index

			if start_index < sample_indices[0]:
				start_index = 0
				stop_index = 0


		return start_index, stop_index


	#Property definition

	@property
	def df(self):
		return self._df
	@df.setter
	def df(self, df):
		self._df = df

	@property
	def X(self):
		return self.X
	@X.setter
	def X(self, X):
		self.X = X

	@property
	def y(self):
		return self._y
	@y.setter
	def df(self, y):
		self._y = y

	@property
	def start_time(self):
		return self._start_time
	@start_time.setter
	def start_time(self,start_time):
		self._start_time = start_time

	@property
	def sqlsession(self):
		return self._sqlsession
	@sqlsession.setter
	def sqlsession(self,sqlsession):
		self._sqlsession = sqlsession
