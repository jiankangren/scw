# Train a Bayesian LSTM on the IMDB sentiment classification task.
# To use the GPU:
#     THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm_regression.py
# To speed up Theano, create a ram disk: 
#     mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then add flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk'

from __future__ import absolute_import
from __future__ import print_function

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import sys
from keras.models import model_from_json
import seaborn as sns
from callbacks import ModelTest
import json
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.regularizers import l2
import matplotlib.pyplot as plt
from pandas import read_csv
import argparse

def parse_args():
	parser = argparse.ArgumentParser("PDR prediction experiment using LSTM network")
	# Environment
	parser.add_argument("--exp-name", type=str, default=None, help="name of the training experiment")
	parser.add_argument("--batch-size", type=int, default=500, help="number of sequences to optimize at the same time")
	parser.add_argument("--look-back", type=int, default=200, help="look back window size used for LSTM sequences")
	parser.add_argument("--look-ahead", type=int, default=200, help="look ahead window size used for PDR calculations")
	parser.add_argument("--mode",type=int, default=0, help="0: use current pos only. 1: use current posand vel. 2: use current pos, vel, CSI. 3: use current pos, vel, CSI, future pos and vel")
	# Training
	parser.add_argument("--training-input", type=str, default=None, help="name of the training input data file")
	parser.add_argument("--training_dir", type=str, default="../dataset/", help="directory in which training data exists")
	parser.add_argument("--epoch-count", type=int, default=1, help="number of epochs used for training")
	parser.add_argument("--save-dir", type=str, default="../model/", help="directory in which training state and model should be saved")
	parser.add_argument("--save-rate", type=int, default=1, help="save model once every time this many epoch are completed")
	parser.add_argument("--lstm-size", type=int, default=64, help="Size of the LSTM hidden layer")
	parser.add_argument("--load-dir", type=str, default="../model/", help="directory in which training state and model are loaded")
	parser.add_argument("--dropout", type=float, default=0.25, help="percentage of dropout inside LSTM model")
	parser.add_argument("--training-percentage", type=float, default=0.8, help="Percentage of the data to be used for training")

	# Evaluation
	parser.add_argument("--evaluate", action="store_true", default=False)
	parser.add_argument("--plot", action="store_true", default=False)
	parser.add_argument("--plots-dir", type=str, default="../figs/", help="directory where plot data is saved")
	return parser.parse_args()

# convert an array of values into a dataset matrix
def create_dataset(dataset_input, dataset_output, look_back=1):
	dataX, dataY = [], []
	for i in range(look_back, len(dataset_input)):
		a = dataset_input[i - look_back:i, :]
		dataX.append(a)
		dataY.append(dataset_output[i, 0])
	return np.array(dataX), np.array(dataY)

# trim the training and testing datasets to have entries multiple of batch_size
def trim_to_batch_size(dataset, batch_size):
	return dataset[0:len(dataset)-(len(dataset)%batch_size)]


def train(arglist):
	# fix random seed for reproducibility
	np.random.seed(7)
	p_emb = 0.25
	p_W = 0.25
	p_U = 0.25

	weight_decay = 1e-4
	batch_size = int(arglist.batch_size)

	# Load data:

	print("Loading data...")
	dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=range(0,(180+5*(arglist.look_ahead+1))),header=None, sep=',', engine='python')
	input_size = (180+5*(arglist.look_ahead+1))
	datasetInput = dataframe.values
	datasetInput = datasetInput.astype('float32')

	# load the Output dataset
	dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[180+5*(arglist.look_ahead+1)],header=None, engine='python')
	datasetOutput = dataframe.values
	datasetOutput = datasetOutput.astype('float32')

	look_back = arglist.look_back
	batch_size = arglist.batch_size
	datasetInput, datasetOutput = create_dataset(datasetInput, datasetOutput, look_back)

	# split into train and test sets
	train_size_input = int(len(datasetInput) * arglist.training_percentage)
	test_size_input = len(datasetInput) - train_size_input
	train_size_output = int(len(datasetOutput) * arglist.training_percentage)
	test_size_output = len(datasetOutput) - train_size_output
	trainX, testX = datasetInput[0:train_size_input,:], datasetInput[train_size_input:len(datasetInput),:]
	trainY, testY = datasetOutput[0:train_size_output], datasetOutput[train_size_output:len(datasetOutput)]
	# Trim data to batch size
	X_train = trim_to_batch_size(trainX, batch_size)
	Y_train = trim_to_batch_size(trainY, batch_size)
	X_test = trim_to_batch_size(testX, batch_size)
	Y_test = trim_to_batch_size(testY, batch_size)

	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], input_size))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], input_size))

	mean_y_train, std_y_train = np.mean(Y_train), np.std(Y_train)

	# Build model:
	print('Build model...')
	#check if there a saved model exists
	modelLoadFilePath= arglist.load_dir + arglist.exp_name + '_model.json'
	if not os.path.isfile(modelLoadFilePath):
		# create and fit the LSTM network
		model = Sequential()
		model.add(LSTM(arglist.lstm_size, dropout=arglist.dropout, batch_input_shape=(batch_size, look_back, input_size)))
		model.add(Dropout(arglist.dropout))
		model.add(Dense(1, activation='sigmoid'))
		data = {}
		# stores loss values over training episodes
		data['loss']  = []
		data['train_score'] = []
		data['test_score'] = []
		#data['val_loss'] = []
		print("Created model")
	else:
		# load json and create model
		json_file = open(modelLoadFilePath, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		# load weights into new model
		model.load_weights(arglist.load_dir + arglist.exp_name + '_model.h5')
		# load data
		#with open(arglist.save_dir + arglist.exp_name + '_data.json') as json_file:
		#	data = json.load(json_file)
		print("Loaded model from disk")

	model.compile(loss='mean_squared_error', optimizer='adam')

	modeltest_1 = ModelTest(X_train[:batch_size],
						 mean_y_train + std_y_train * np.atleast_2d(Y_train[:batch_size]).T,
						 test_every_X_epochs=1, verbose=0, loss='euclidean',
						 mean_y_train=mean_y_train, std_y_train=std_y_train, batch_size=arglist.batch_size, arglist=arglist)
	tensorflow_test_size = batch_size * (len(X_test) / batch_size)
	modeltest_2 = ModelTest(X_test, np.atleast_2d(Y_test).T,
						 test_every_X_epochs=1, verbose=0, loss='euclidean',
						 mean_y_train=mean_y_train, std_y_train=std_y_train, batch_size=arglist.batch_size, arglist=arglist)
	#tensorflow_train_size = batch_size * (len(X_train) / batch_size)
	model.fit(X_train, Y_train,
		   batch_size=arglist.batch_size, epochs=arglist.epoch_count, callbacks=[modeltest_1, modeltest_2])

	# Theano
	#modeltest_1 = ModelTest(X_train,
	#						mean_y_train + std_y_train * np.atleast_2d(Y_train).T,
	#						test_every_X_epochs=1, verbose=0, loss='euclidean', batch_size=arglist.batch_size,
	#						mean_y_train=mean_y_train, std_y_train=std_y_train, arglist=arglist)
	#modeltest_2 = ModelTest(X_test, np.atleast_2d(Y_test).T, test_every_X_epochs=1,
#							verbose=0, loss='euclidean', batch_size=arglist.batch_size,
#							mean_y_train=mean_y_train, std_y_train=std_y_train, arglist=arglist)
#	model.fit(X_train, Y_train, batch_size=batch_size, epochs=arglist.epoch_count,
#			  callbacks=[modeltest_1, modeltest_2])

	#standard_prob = model.predict(X_train, batch_size=500, verbose=1)
	#print(np.mean(((mean_y_train + std_y_train * np.atleast_2d(Y_train).T)
	#			   - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)


	# Dropout approximation for test data:
	#standard_prob = model.predict(X_test, batch_size=arglist.batch_size, verbose=1)
	#print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)

	# MC dropout for test data:
	T = 50
	prob_train = np.array([modeltest_1.predict_stochastic(X_train, batch_size=arglist.batch_size, verbose=0)
					 for _ in range(T)])
	prob_mean_train = np.mean(prob_train, 0)
	prob_std_train = np.std(prob_train,0)
	print('Train RMSE:' , np.mean((np.atleast_2d(Y_train).T - (prob_mean_train))**2, 0)**0.5)
	print('Train Abs(mean-std)', np.mean(np.abs(prob_mean_train-prob_std_train)))
	#mean_y_train + std_y_train *
	# MC dropout for test data:
	prob_test = np.array([modeltest_2.predict_stochastic(X_test, batch_size=arglist.batch_size, verbose=0)
					 for _ in range(T)])
	prob_mean_test = np.mean(prob_test, 0)
	prob_std_test = np.std(prob_test,0)
	print('Test RMSE:' ,np.mean((np.atleast_2d(Y_test).T - (prob_mean_test))**2, 0)**0.5)
	print('Test Abs(mean-std)', np.mean(np.abs(prob_mean_test-prob_std_test)))
	yy1_train = 100 * np.reshape(prob_mean_train+prob_std_train,len(prob_mean_train))
	yy2_train = 100 * np.reshape(prob_mean_train-prob_std_train,len(prob_mean_train))
	yy3_train = 100 * np.reshape(prob_mean_train+2*prob_std_train,len(prob_mean_train))
	yy4_train = 100 * np.reshape(prob_mean_train-2*prob_std_train,len(prob_mean_train))
	yy5_train = 100 * np.reshape(prob_mean_train+3*prob_std_train,len(prob_mean_train))
	yy6_train = 100 * np.reshape(prob_mean_train-3*prob_std_train,len(prob_mean_train))
	yy1_test = 100 * np.reshape(prob_mean_test+prob_std_test,len(prob_mean_test))
	yy2_test = 100 * np.reshape(prob_mean_test-prob_std_test,len(prob_mean_test))
	yy3_test = 100 * np.reshape(prob_mean_test+2*prob_std_test,len(prob_mean_test))
	yy4_test = 100 * np.reshape(prob_mean_test-2*prob_std_test,len(prob_mean_test))
	yy5_test = 100 * np.reshape(prob_mean_test+3*prob_std_test,len(prob_mean_test))
	yy6_test = 100 * np.reshape(prob_mean_test-3*prob_std_test,len(prob_mean_test))
	if arglist.plot == True:
		# plot prediction vs groundtruth of training data
		time = 0.02 * np.arange(0, len(X_train))
		plt.figure()
		plt.xlabel('Time (sec)')
		plt.ylabel('PDR (%)')
		plt.plot(time, 100 *prob_mean_train,color='black')
		plt.fill_between(time,yy1_train,yy2_train,color='gray')
		plt.fill_between(time,yy1_train,yy3_train,color='gray', alpha=0.75)
		plt.fill_between(time,yy2_train,yy4_train,color='gray', alpha=0.75)
		plt.fill_between(time,yy3_train,yy5_train,color='gray', alpha=0.4)
		plt.fill_between(time,yy4_train,yy6_train,color='gray', alpha=0.4)
		plt.plot(time, 100*Y_train,color='red')
		#plt.plot(time, 100 *(prob_mean_train+prob_std_train))
		#plt.plot(time, 100 *(prob_mean_train-prob_std_train))
		plt.legend(['Prediction', 'Ground Truth'])
		plt.show()

		# plot prediction vs groundtruth of testing data
		time = 0.02 * np.arange(0, len(X_test))
		plt.figure()
		plt.xlabel('Time (sec)')
		plt.ylabel('PDR (%)')
		plt.plot(time, 100 *prob_mean_test, color='black')
		#plt.plot(time, 100 *(prob_mean_test+prob_std_test))
		#plt.plot(time, 100 *(prob_mean_test-prob_std_test))
		plt.fill_between(time,yy1_test,yy2_test,color='gray')
		plt.fill_between(time,yy1_test,yy3_test,color='gray', alpha=0.75)
		plt.fill_between(time,yy2_test,yy4_test,color='gray', alpha=0.75)
		plt.fill_between(time,yy3_test,yy5_test,color='gray', alpha=0.4)
		plt.fill_between(time,yy4_test,yy6_test,color='gray', alpha=0.4)
		plt.plot(time, 100*Y_test, color='red')
		plt.legend(['Prediction', 'Ground Truth'])
		plt.show()

		# plot prediction vs groundtruth of testing and training data
		time = 0.02 * np.arange(0, len(X_train) + len(X_test))
		plt.figure()
		plt.xlabel('Time (sec)')
		plt.ylabel('PDR (%)')
		plt.plot(time, 100*np.concatenate((Y_train, Y_test), axis=0))
		plt.plot(time, 100 * np.concatenate((prob_mean_train, prob_mean_test), axis=0))
		plt.plot(time, 100 * np.concatenate((prob_mean_train+prob_std_train,prob_mean_test+prob_std_test), axis=0))
		plt.plot(time, 100 * np.concatenate((prob_mean_train-prob_std_train,prob_mean_test-prob_std_test), axis=0))
		plt.legend(['Ground Truth', 'Prediction'])
		plt.show()

if __name__ == '__main__':
	arglist = parse_args()
	train(arglist)
