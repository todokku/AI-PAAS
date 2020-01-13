import numpy as np
import random
from functools import reduce

from sklearn.model_selection import train_test_split

class GridDataHandler():

	'''
	TODO: column information here
	'''

	# Method definition

	def __init__(self, data_scaler=None):
		# Public properties

		# Splitting. This is what is used to train
		self._X_train = None
		self._X_crossVal = None
		self._X_test = None
		self._y_train = None
		self._y_crossVal = None
		self._y_test = None

		self._data_scaler = data_scaler

	# Public
	def load_data(self, verbose = 0, cross_validation_ratio = 0, unroll=False, **kwargs):
		"""Unroll just to keep compatibility with the API"""

		boundaries = kwargs['boundaries']  #Boundaries for each of the dimensions of the problem. Is a list of lists
		n = kwargs['n'] #Number of points between the upper and lower bound for each of the dimensions of the problem.

		if verbose == 1:
			print("Loading data. Cros-Validation ratio {}".format(cross_validation_ratio))

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		self._X_train, self._y_train = self.grid_sample(boundaries, n)


		if self._data_scaler != None:
			self._X_train = self._data_scaler.fit_transform(self._X_train)

		#Test data is 10% of entire data
		self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X_train, self._y_train, train_size=0.9)

		#Create cross-validation
		if cross_validation_ratio != 0:
			self._X_train, self._X_crossVal, self._y_train, self._y_crossVal = train_test_split(self._X_train, self._y_train, train_size=1-cross_validation_ratio)


	def grid_sample(self, boundaries, n_points):
		"""For each dimension sample n_points points from boundaries[0] to boundaries[1]. Build a d-dimensional grid (Z) with these points"""


		n_dimensions = len(boundaries)
		x = [np.linspace(boundaries[i][0], boundaries[i][1], n_points[i]) for i in range(n_dimensions)]
		indices_counter = np.zeros([n_dimensions,], dtype=np.int32)
		indices_limits = np.array(n_points, dtype=np.int32)

		n_rows =  reduce(lambda x,y: x*y, n_points) if len(n_points) > 1 else n_points[0]
		#X = np.zeros(shape=[n_rows, n_dimensions])
		X = []

		indices_sum = np.sum((indices_limits-1) - indices_counter)

		#Do until every point cartesian product is complete
		dimension_indices = list(range(n_dimensions))

		#Perform the cartesian product with each dimension
		while(indices_sum > 0):
			row = [x[i][j] for i,j in zip(dimension_indices, indices_counter)]
			X.append(row)

			# Check if every point in the arrays has been considered.
			indices_sum = np.sum((indices_limits - 1) - indices_counter)
			if indices_sum != 0:

				#Update counter for each dimension
				update_complete = False
				index_update_index = n_dimensions - 1
				while update_complete != True:
					update_index = indices_counter[index_update_index]
					update_index = update_index + 1

					if update_index < indices_limits[index_update_index]:
						indices_counter[index_update_index] = update_index
						update_complete = True
					else: #Reset current dimension index to 0 and go to next dimension
						indices_counter[index_update_index] = 0
						index_update_index = index_update_index - 1

		Z = np.array(X)
		y = np.zeros([Z.shape[0], 1])

		return Z, y


	def print_data(self, print_top=True):
		"""Print the shapes of the data and the first 5 rows"""

		if self._X_train is None:
			print("No data available")
			return

		print("Printing shapes\n")

		print("Training data (X, y)")
		print(self._X_train.shape)
		print(self._y_train.shape)

		if self._X_crossVal is not None:
			print("Cross-Validation data (X, y)")
			print(self._X_crossVal.shape)
			print(self._y_crossVal.shape)

		print("Testing data (X, y)")
		print(self._X_test.shape)
		print(self._y_test.shape)

		if print_top == True:
			print("Printing first 5 elements\n")

			print("Training data (X, y)")
			print(self._X_train[:5])
			print(self._y_train[:5])

			if self._X_crossVal is not None:
				print("Cross-Validation data (X, y)")
				print(self._X_crossVal[:5])
				print(self._y_crossVal[:5])

			print("Testing data (X, y)")
			print(self._X_test[:5])
			print(self._y_test[:5])
		else:
			print("Printing last 5 elements\n")

			print("Training data (X, y)")
			print(self._X_train[-5:])
			print(self._y_train[-5:])

			if self._X_crossVal is not None:
				print("Cross-Validation data (X, y)")
				print(self._X_crossVal[-5:])
				print(self._y_crossVal[-5:])

			print("Testing data (X, y)")
			print(self._X_test[-5:])
			print(self._y_test[-5:])

	# Property definition

	@property
	def X_train(self):
		return self._X_train

	@X_train.setter
	def X_train(self, X_train):
		self._X_train = X_train

	@property
	def X_crossVal(self):
		return self._X_crossVal

	@X_crossVal.setter
	def X_crossVal(self, X_crossVal):
		self._X_crossVal = X_crossVal

	@property
	def X_test(self):
		return self._X_test

	@X_test.setter
	def X_test(self, X_test):
		self._X_test = X_test

	@property
	def y_train(self):
		return self._y_train

	@y_train.setter
	def y_train(self, y_train):
		self._y_train = y_train

	@property
	def y_crossVal(self):
		return self._y_crossVal

	@y_crossVal.setter
	def y_crossVal(self, y_crossVal):
		self._y_crossVal = y_crossVal

	@property
	def y_test(self):
		return self._y_test

	@y_test.setter
	def y_test(self, y_test):
		self._y_test = y_test


