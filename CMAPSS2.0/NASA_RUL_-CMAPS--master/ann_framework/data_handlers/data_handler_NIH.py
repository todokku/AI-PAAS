import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
from itertools import chain
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


class NIHDataHandler():

	def __init__(self, data_file, data_folder, data_scaler=None, multi_labels=True):
		#ReadOnly properties
		
		self._data_file = data_file
		self._data_folder = data_folder
		self._all_df = None
		self._all_labels = None


		self._X_train = None
		self._y_train = None
		"""
		self._X_crossVal = None
		self._y_crossVal = None
		"""
		self._X_test = None
		self._y_test = None
		
		self._train_gen = None
		self._valid_gen = None
		self._data_scaler = data_scaler

		self._multi_labels = multi_labels
		
        
	def load_csv_into_df(self): #file_name = data_file
		"""Given the filename, load the data into a pandas dataframe"""

		df = pd.read_csv(self.data_file)

		#drop last column because it is null
		df = df.iloc[:,:-1]
        
		all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join(self._data_folder, 'images*', '*', '*.png'))}

		#print('all_image_paths',all_image_paths)
		print('Scans found:', len(all_image_paths), ', Total Headers', df.shape[0])
		df['path'] = df['Image Index'].map(all_image_paths.get)
		df['Patient Age'] = df['Patient Age'].map(lambda x: int(x))
        
		return df
	

	def generate_labels(self):
		"""Add columns to the dataset containing binary labels for each of the """

		#label_counts = xray_df['Finding Labels'].value_counts()[:15] 
        
		self._all_df['Finding Labels'] = self._all_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
		self._all_labels = np.unique(list(chain(*self._all_df['Finding Labels'].map(lambda x: x.split('|')).tolist()))) #Find all unique labels
		self._all_labels = [x for x in self._all_labels if len(x)>0] #Discard no findings label since was replaced by ''
		#print('All Labels ({}): {}'.format(len(all_labels), all_labels))

		#Create one hot encoding for each image (in the dataframe)
		for c_label in self._all_labels:
			if len(c_label)>1: # leave out empty labels
				self._all_df[c_label] = self.all_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
				

	def resample_set(self, number_samples, weights=None):

		if weights == None:
			weights = self.all_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2 #Add small number to avoid probability 0
			weights /= weights.sum()

		self._all_df = self.all_df.sample(number_samples, weights=weights) 

		
	def keep_n_diseases(self, patient):

		disease_vector = patient[self._all_labels]
		positive_diseases = np.nonzero(disease_vector)[0]
		
		if len(positive_diseases) > 1:
			disease_vector[positive_diseases[1:]] = 0

		return disease_vector
		
		
	def load_data(self, verbose = 0, cross_validation_ratio = 0, unroll=None, prune_threshold=0, number_samples=0, resample_weights=None):    
		"""Load the data"""

		self.all_df = self.load_csv_into_df()
		
		self.generate_labels()
		#print(self._all_df.head())
		#print(self._all_labels)
		#optional function here
		
		if prune_threshold > 0:
			#Prune labels with less than prune_threshold elements
			self._all_labels = [c_label for c_label in self._all_labels if self._all_df[c_label].sum()>prune_threshold]

		#Get number_samples elements with a distribution defined by weights
		if number_samples != 0:
			self.resample_set(number_samples, weights=resample_weights)


		#Generate label vector from all the diseases column
		self._all_df['disease_vec'] = self._all_df.apply(lambda x: [x[self._all_labels].values], 1).map(lambda x: x[0])

		#self._all_df[self._all_labels] = self._all_df.apply(lambda x : np.nonzero(x[self._all_labels].values == 1)[0], 1).map(lambda x: print(x))

		#Keep only the first encountered disease
		if self.multi_labels == False:
			self._all_df[self._all_labels] = self._all_df.apply(self.keep_n_diseases, 1)

		#Generate label vector from all the diseases column
		self._all_df['disease_vec_n_only'] = self._all_df.apply(lambda x: [x[self._all_labels].values], 1).map(lambda x: x[0])

		print(self._all_df)
		
		#Remove random_state after testing
		#Stratification is done only considering the first finding per image
		train_df, valid_df = train_test_split(self._all_df, test_size = cross_validation_ratio,
                                   stratify = self._all_df['Finding Labels'].map(lambda x: x[:4]))
		
		#print('train', train_df.shape[0], 'validation', valid_df.shape[0])
		IMG_SIZE, core_idg =  self.generate_data()
		#print(train_df['path'])
		#print(self._all_df['disease_vec'].iloc[0])
		#print(type(self._all_df['disease_vec'].iloc[0]))

		train_batch_size = 32
		cv_batch_size = 256
		test_batch_size = 1024

		#Get the iterator for the train set
		self._train_gen = core_idg.flow_from_dataframe(train_df,directory=None, x_col='path', y_col=self._all_labels, target_size = IMG_SIZE,
							       class_mode='other', color_mode='grayscale', batch_size=train_batch_size)

		"""
		self._train_gen2 = self.flow_from_dataframe(img_data_gen = core_idg, in_df = train_df, path_col = 'path',
							   y_col='disease_vec', target_size = IMG_SIZE, color_mode = 'grayscale',
							    batch_size = train_batch_size) 
		"""
		#Get the iterator for the cv set
		self._valid_gen = core_idg.flow_from_dataframe(valid_df, directory=None, x_col='path', y_col=self._all_labels, target_size = IMG_SIZE,
							       class_mode='other', color_mode='grayscale', batch_size=cv_batch_size)
		"""
		self._valid_gen2 = self.flow_from_dataframe(core_idg, valid_df, path_col = 'path', y_col = 'disease_vec', target_size = IMG_SIZE,
							   color_mode = 'grayscale', batch_size = cv_batch_size) # we can use much larger batches for evaluation
		"""
		# used a fixed dataset for evaluating the algorithm
		self._X_test, self._y_test = next(core_idg.flow_from_dataframe(valid_df, directory=None, x_col = 'path', y_col=self._all_labels,
									       target_size = IMG_SIZE, color_mode = 'grayscale',
									       class_mode='other', batch_size = test_batch_size)) # one big batch

		"""
		self._X_test2, self._y_test2 = next(self.flow_from_dataframe(core_idg, valid_df, path_col='path', y_col='disease_vec',
									   target_size = IMG_SIZE, color_mode = 'grayscale', batch_size = test_batch_size)) # one big batch
		"""                                      

		
	def generate_data(self):
		"""Create the data generator, rather than performing the operations on the entire dataset in memory
		the class is designed to be iterated by deep learning model fitting process, creating augmented data
		just in time. The data generator itself is in fact a generator, returning batches of 
		image samples when requested. Batches are obtained through the flow function."""
		
		IMG_SIZE = (128, 128)
		core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)
		return IMG_SIZE, core_idg
	

	def flow_from_dataframe(self,img_data_gen, in_df, path_col, y_col, **dflow_args):
		base_dir = os.path.dirname(in_df['path'].values[0])
		print('## Ignore next message from keras, values are replaced anyways')
		df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
		df_gen.filenames = in_df[path_col].values
		df_gen.classes = np.stack(in_df[y_col].values)
		df_gen.samples = in_df.shape[0]
		df_gen.n = in_df.shape[0]
		df_gen._set_index_array()
		df_gen.directory = '' # since we have the full path
		print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
		return df_gen

	
	def print_data(self, print_top=True):
		"""Print the shapes of the data and the first 5 rows"""

		if self._train_gen is None:
			print("No data available")
			return

		print("Printing shapes\n")

		sample_X_train, sample_y_train = next(self._train_gen)
		sample_X_cv, sample_y_cv = next(self._valid_gen) 
		
		print("Training data (X, y)")
		print(sample_X_train.shape)
		print(sample_y_train.shape)
		
		if sample_X_cv is not None:
			print("Cross-Validation data (X, y)")
			print(sample_X_cv.shape)
			print(sample_y_cv.shape)

		print("Testing data (X, y)")
		print(self._X_test.shape)
		print(self._y_test.shape)

		if print_top == True:
			print("Printing first 5 elements\n")
			
			print("Training data (X, y)")
			print(sample_X_train[:5,:])
			print(sample_y_train[:5,:])

			if sample_X_cv  is not None:
				print("Cross-Validation data (X, y)")
				print(sample_X_cv [:5,:])
				print(sample_y_cv[:5,:])

			print("Testing data (X, y)")
			print(self._X_test[:5,:])
			print(self._y_test[:5,:])
		else:
			print("Printing last 5 elements\n")
			
			print("Training data (X, y)")
			print(sample_X_train[-5:,:])
			print(sample_y_train[-5:,:])

			if sample_X_cv  is not None:
				print("Cross-Validation data (X, y)")
				print(sample_X_cv [-5:,:])
				print(sample_y_cv[-5:,:])

			print("Testing data (X, y)")
			print(self._X_test[-5:,:])
			print(self._y_test[-5:,:])
            
	@property
	def data_file(self):
		return self._data_file

	@data_file.setter
	def data_file(self, data_file):
		self._data_file = data_file

	@property
	def data_folder(self):
		return self._data_folder

	@data_folder.setter
	def data_folder(self, data_folder):
		self._data_folder = data_folder
      
	@property
	def all_df(self):
		return self._all_df

	@all_df.setter
	def all_df(self, all_df):
		self._all_df = all_df
        
	@property
	def all_labels(self):
		return self._all_labels

	@all_labels.setter
	def all_labels(self, all_labels):
		self._all_labels = all_labels

	"""	
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
	"""
            
	@property
	def X_test(self):
		return self._X_test

	@X_test.setter
	def X_test(self, X_test):
		self._X_test = X_test
            
	@property
	def train_gen(self):
		return self._train_gen

	@train_gen.setter
	def train_gen(self, train_gen):
		self._train_gen = train_gen   

	@property
	def valid_gen(self):
		return self._valid_gen

	@valid_gen.setter
	def valid_gen(self, valid_gen):
		self._valid_gen = valid_gen          
            
	@property
	def data_scaler(self):
		return self._data_scaler

	@data_scaler.setter
	def data_scaler(self, data_scaler):
		self._data_scaler = data_scaler  

	@property
	def normalize(self):
		return self._normalize

	@normalize.setter
	def normalize(self, normalize):
		self._normalize = normalize

	@property
	def multi_labels(self):
		return self._multi_labels

	@multi_labels.setter
	def multi_labels(self, multi_labels):
		self._multi_labels = multi_labels        

	"""
	@property
	def y_train(self):
		return self._y_train
    
	@property
	def y_crossVal(self):
		return self._y_crossVal
	"""
    
	@property
	def y_test(self):
		return self._y_test    
            
        
