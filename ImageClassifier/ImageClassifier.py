import time
import gzip, os
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Dropout,MaxPooling2D

import utils

class MyCallback(tf.keras.callbacks.Callback):
	"""
		Custome callback to calculate training ETA
	"""
	def __init__(self, epochs):
		super().__init__()
		self.epochs = epochs
		
	def on_epoch_begin(self, epoch, logs=None):
		if epoch == 0:
			self.start_time = time.time()
			print("Epoch 1 in progress..", end='\r')
		else:
			time_elapsed = time.time() - self.start_time
			eta = (time_elapsed/epoch) * (self.epochs-epoch)/60.0
			print("Training ETA : %.1f min(s)" %eta, end='\r')
	
	def on_train_end(self, logs=None):
		print("Training Completed..               ")


class ImageClassifier(object):
	"""
		Class carrying all the functionalities for CNN based image classification 
		and prediction
	"""
	def __init__(self, img_dim=(28, 28)):
		"""
			Parameter(s):
				img_dim : (width, height)
		"""
		assert len(img_dim)==2, "Image dimensions should be in form (width, height)"
		self.IMG_W, self.IMG_H = img_dim
		self._cnn_model = None
	
	
	def _preprocess_data(self, images_arr, labels_arr=None):
		"""
			Take the images and labels data and return the scaled and reshaped form

			Parameter(s):
				images_arr : list of images data of shape (samples,img_h,img_w)
				labels_arr : list of labels data of shape (samples,)
		"""
		images_arr = images_arr.reshape(-1, self.IMG_H, self.IMG_W, 1)
		images_arr = (255.0 - images_arr) / 255.0
		# images_arr = images_arr / 255.0
		if labels_arr is not None:
			labels_arr = to_categorical(labels_arr, num_classes=len(set(labels_arr)))
		return images_arr, labels_arr
	
	
	def load_from_raw_data(self, train_images_path, train_labels_path, test_images_path, test_labels_path,
										 class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 
																		'Shirt', 'Sneaker', 'Bag', 'Ankle boot']):
		"""
			Load image and labels data in pixel value format. Returns the train, validation and 
			test set, after preprocessing.
			
			Parameter(s):
				train_images_path, train_labels_path, test_images_path, test_labels_path
				class_labels : labels for each class values
		"""
		assert os.path.isfile(train_images_path), "Images training data missing"
		assert os.path.isfile(train_labels_path), "Labels training data missing"
		assert os.path.isfile(test_images_path), "Images test data missing"
		assert os.path.isfile(test_labels_path), "Labels test data missing"
		
		# Load images
		train_images = utils.load_image_data(train_images_path).reshape(-1, self.IMG_H, self.IMG_W)
		test_images = utils.load_image_data(test_images_path).reshape(-1, self.IMG_H, self.IMG_W)
		
		# Load labels
		train_labels = utils.load_label_data(train_labels_path)
		test_labels = utils.load_label_data(test_labels_path)
		
		# Split training data for training and validation
		train_images, val_images, train_labels, val_labels = train_test_split(train_images, 
																					train_labels, test_size=0.25, random_state=123)
		
		# Preprocess the image data by scaling and categorizing
		self._train_images, self._train_labels = self._preprocess_data(train_images, train_labels)
		self._val_images, self._val_labels = self._preprocess_data(val_images, val_labels)
		self._test_images, self._test_labels = self._preprocess_data(test_images, test_labels)
		self._class_labels = class_labels
		print("Data loaded successfully.")
	
	
	def load_from_image_files(self, data_dir):
		"""
			Load image and labels data from raw images, where each class of images are kept in
			their respective directories, having directory name as class names. 
			Returns the train, validation and test set, after preprocessing.
			
			Parameter(s):
				data_dir : folder of images data
		"""
		assert os.path.isdir(data_dir), "Image directory is not valid"
		
		images_arr, labels_arr, class_labels = list(), list(), list()
		for fi,folder in enumerate(os.listdir(data_dir)):
			success, total = 0, 0
			class_labels.append(folder.title())
			for file in os.listdir(os.path.join(data_dir, folder)):
				try:
					img_arr = utils.process_image(os.path.join(data_dir,folder,file))
					images_arr.append(img_arr)
					labels_arr.append(fi)
					success += 1
				except:
					print("Error in processing file: %s" %(os.path.join(data_dir,folder,file)))
				finally:
					total += 1
			print("%d out of %d images processed for label - %s" %(success, total, folder.title()))
		self._class_labels = class_labels
		
		images_arr, labels_arr = np.array(images_arr), np.array(labels_arr)
		# Split training data for training, validation and testing
		train_images, val_images, train_labels, val_labels = train_test_split(images_arr, 
																	labels_arr, test_size=0.30, stratify=labels_arr, random_state=123)
		val_images, test_images, val_labels, test_labels = train_test_split(val_images, 
																	val_labels, test_size=0.30, stratify=val_labels, random_state=123)
		
		# Re-shape
		train_images = train_images.reshape(-1, self.IMG_H, self.IMG_W)
		val_images = val_images.reshape(-1, self.IMG_H, self.IMG_W)
		test_images = test_images.reshape(-1, self.IMG_H, self.IMG_W)
		
		# Preprocess the image data by scaling and categorizing
		self._train_images, self._train_labels = self._preprocess_data(train_images, train_labels)
		self._val_images, self._val_labels = self._preprocess_data(val_images, val_labels)
		self._test_images, self._test_labels = self._preprocess_data(test_images, test_labels)
		print("Data loaded successfully.")
	
		
	def _create_cnn_model(self):
		"""
			Defines and returns the CNN model
		"""
		cnn_model = Sequential()
		cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', 
																								 input_shape=(self.IMG_W, self.IMG_H, 1)))
		cnn_model.add(MaxPooling2D((2, 2)))
		cnn_model.add(Dropout(0.25))
		cnn_model.add(Flatten())
		cnn_model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
		cnn_model.add(Dense(len(self._class_labels), activation='softmax'))
		
		cnn_model.compile(optimizer='adam',
											loss='categorical_crossentropy',
											metrics=['accuracy'])
		return cnn_model
	
	
	def train(self, save_dir='./model'):
		"""
			Train the CNN model
			
			Parameter(s):
				save_dir : directory where you want the model to be saved after training. 
										Pass None if you want to train just for checking
		
		"""
		assert self._class_labels is not None, "Class labels not defined"
		
		self._cnn_model = self._create_cnn_model()
		print("Training initiated..")
		history = self._cnn_model.fit(self._train_images, self._train_labels, batch_size=256, 
																 epochs=10, verbose=0, callbacks=[MyCallback(10)],
																 validation_data=(self._val_images, self._val_labels))
		
		# Print and save history if specified
		print("Training Loss : %.3f" % history.history['loss'][-1])
		print("Validation Loss : %.3f" % history.history['val_loss'][-1])
		print("Training Accuracy : %.3f" % history.history['accuracy'][-1])
		print("Validation Accuracy : %.3f" % history.history['val_accuracy'][-1])
		
		if save_dir is not None:
			utils.save_run_history(history)
			self._cnn_model.save(save_dir)
		
		score = self._cnn_model.evaluate(self._test_images, self._test_labels, verbose=0)
		print("Test Loss : %.3f" % score[0])
		print("Test Accuracy : %.3f" % score[1])
		
		if save_dir is not None:
			y_true = self._test_labels.argmax(axis=1)
			y_pred = np.argmax(self._cnn_model.predict(self._test_images), axis=-1)
			utils.save_confusion_matrix(y_true, y_pred, self._class_labels)
			utils.save_classification_report(y_true, y_pred, self._class_labels)
	
	
	def predict_from_raw_data(self, images_path):
		"""
			Load image data given in pixel value format from the file and return the predicted labels.
			
			Parameter(s):
				images_path
		"""
		assert os.path.isfile(images_path), "Images data file missing"
		assert self._cnn_model is not None, "Model training has not been done"
		
		# Load images
		images_data = utils.load_image_data(images_path).reshape(-1, self.IMG_H, self.IMG_W)
		images_data,_ = self._preprocess_data(images_data)
		classes = np.argmax(self._cnn_model.predict(images_data), axis=-1)
		labels = list(map(lambda i:self._class_labels[i], classes))
		return labels
	
	
	def predict_from_image_files(self, data_dir):
		"""
			Load image data from the image files and return the predicted labels.
			
			Parameter(s):
				data_dir : location of images
		"""
		assert os.path.isdir(data_dir), "Image directory is not valid"
		assert self._cnn_model is not None, "Model training has not been done"
		
		error_index, images_arr = dict(), list()
		for fi,file in enumerate(os.listdir(data_dir)):
			try:
				img_arr = utils.process_image(os.path.join(data_dir,file))
				images_arr.append(img_arr)
			except Exception as ex:
				print("Error in processing file: %s : %s" %(os.path.join(data_dir,file), str(ex)))
				error_index[fi] = True
		
		images_arr = np.array(images_arr).reshape(-1, self.IMG_H, self.IMG_W)
		images_arr,_ = self._preprocess_data(images_arr)
		classes = np.argmax(self._cnn_model.predict(images_arr), axis=-1).tolist()
		
		# prepare labels from classes
		labels = list()
		for ci in range(len(classes)+len(error_index)):
			if error_index.get(ci) is not None:
				labels.append('Unknown')
			else:
				labels.append(self._class_labels[classes.pop(0)])
		return labels
	
	
	def load_trained_model(self, model_dir='./model/', class_labels = ['T-shirt/top', 'Trouser', 
																										 'Pullover', 'Dress', 'Coat', 'Sandal', 
																									 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']):
		"""
			Load image data from the image files and return the predicted labels.
			
			Parameter(s):
				model_dir : directory of saved model
				class_labels : labels of class values
		"""
		try:
			self._cnn_model = tf.keras.models.load_model(model_dir)
			self._class_labels = class_labels
		except:
			print("Model could not be loaded.")    



if __name__ == "__main__":
	img_clf = ImageClassifier()

	# file path for training and test data
	# train_images_path = '../data/fashion/train-images-idx3-ubyte.gz'
	# train_labels_path = '../data/fashion/train-labels-idx1-ubyte.gz'
	# test_images_path = '../data/fashion/t10k-images-idx3-ubyte.gz'
	# test_labels_path = '../data/fashion/t10k-labels-idx1-ubyte.gz'

	# img_clf.load_from_raw_data(train_images_path, train_labels_path, test_images_path, test_labels_path)

	# training 
	# img_clf.train()

	img_clf.load_trained_model()

	print(img_clf.predict_from_image_files('../data/test/'))