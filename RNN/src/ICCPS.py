import argparse
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import os
import os.path
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import CuDNNLSTM
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn import neighbors


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
	parser.add_argument("--epoch-count", type=int, default=30, help="number of epochs used for training")
	parser.add_argument("--save-dir", type=str, default="../model/", help="directory in which training state and model should be saved")
	parser.add_argument("--save-rate", type=int, default=1, help="save model once every time this many epoch are completed")
	parser.add_argument("--lstm-size", type=int, default=64, help="Size of the LSTM hidden layer")
	parser.add_argument("--load-dir", type=str, default="../model/", help="directory in which training state and model are loaded")
	parser.add_argument("--dropout", type=float, default=0, help="percentage of dropout inside LSTM model")
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
	# load the Input dataset
	if arglist.mode ==3:
		dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[*range(0,(180+5*(arglist.look_ahead+1)))],header=None, sep=',', engine='python')
		input_size = (180+5*(arglist.look_ahead+1))
	elif arglist.mode ==2:
		dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[*range(0,5), *range(5*(arglist.look_ahead+1),(180+5*(arglist.look_ahead+1)))],header=None, sep=',', engine='python')
		input_size = 185
	elif arglist.mode ==1:
		dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[*range(0,5)],header=None, sep=',', engine='python')
		input_size = 5
	elif arglist.mode ==0:
		dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[*range(0,2)],header=None, sep=',', engine='python')
		input_size = 2
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
	trainX = trim_to_batch_size(trainX, batch_size)
	trainY = trim_to_batch_size(trainY, batch_size)
	testX = trim_to_batch_size(testX, batch_size)
	testY = trim_to_batch_size(testY, batch_size)
	#trainX, trainY = create_dataset(trainInput, trainOutput, look_back)
	#testX, testY = create_dataset(testInput, testOutput, look_back)
	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], input_size))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], input_size))

	# check if there a saved model exists
	modelLoadFilePath= arglist.load_dir + arglist.exp_name + '_model.json'
	if not os.path.isfile(modelLoadFilePath):
		# create and fit the LSTM network
		model = Sequential()
		#model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
		model.add(CuDNNLSTM(arglist.lstm_size, return_sequences=True, batch_input_shape=(batch_size, look_back, input_size)))
		model.add(CuDNNLSTM(arglist.lstm_size, return_sequences=True, batch_input_shape=(batch_size, look_back, input_size)))
		model.add(CuDNNLSTM(arglist.lstm_size, batch_input_shape=(batch_size, look_back, input_size)))
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
		with open(arglist.save_dir + arglist.exp_name + '_data.json') as json_file:
			data = json.load(json_file)
		print("Loaded model from disk")

	model.compile(loss='mean_squared_error', optimizer='adam')

	for i in range(arglist.epoch_count):
		model_hist = model.fit(trainX, trainY,  epochs=arglist.save_rate, batch_size=arglist.batch_size, verbose=2)
	#   uncomment to use stateful LSTMS
	#	model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2, shuffle=False)
	#	model.reset_states()
		# serialize model to JSON
		model_json = model.to_json()
		with open(arglist.save_dir + arglist.exp_name + '_model.json', "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(arglist.save_dir + arglist.exp_name + '_model.h5')
		print("Saved model to disk")

		# evaluate the model using testing data
		train_score = model.evaluate(trainX, trainY, batch_size=batch_size)
		test_score = model.evaluate(testX, testY, batch_size=batch_size)
		data['loss'].append(model_hist.history['loss'][0])
		data['train_score'].append(train_score)
		data['test_score'].append(test_score)
		with open(arglist.save_dir + arglist.exp_name + '_data.json', "w") as json_file:
			json.dump(data,json_file)
	if arglist.plot == True:
		# uncomment to use stateful LSTMS
		#model.reset_states()
		trainPredict = model.predict(trainX, batch_size=batch_size)
		trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
		print('Train Score: %.2f RMSE' % (trainScore))

		testPredict = model.predict(testX, batch_size=batch_size)
		# calculate root mean squared error
		testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
		print('Test Score: %.2f RMSE' % (testScore))

		# plot results
		## plot prediction vs groundtruth of training data
		#time = 0.02 * np.arange(0, len(trainX))

		#plt.figure()
		#plt.xlabel('Time (sec)')
		#plt.ylabel('PDR (%)')
		#plt.plot(time, 100* trainY)
		#plt.show()

		#plt.figure()
		#plt.xlabel('Time (sec)')
		#plt.ylabel('PDR (%)')
		#plt.plot(time, 100* trainY)
		#plt.plot(time, 100 *trainPredict)
		#plt.legend(['Ground Truth', 'Predictions'])
		#plt.show()
		## plot prediction vs groundtruth of testing data
		#time = 0.02 * np.arange(0, len(testX))
		#plt.figure()
		#plt.xlabel('Time (sec)')
		#plt.ylabel('PDR (%)')
		#plt.plot(time, 100*testY)
		#plt.plot(time, 100 *testPredict)
		#plt.legend(['Ground Truth', 'Predictions'])
		#plt.show()
		## plot loss
		#plt.figure()
		#plt.xlabel('Number of Epochs')
		#plt.ylabel('Training Loss Value')
		#plt.plot(np.arange(0, len(data['loss'])), data['loss'])
		#plt.show()
		## plot loss
		#plt.figure()
		#plt.xlabel('Number of Epochs')
		#plt.ylabel('RMSE')
		#plt.plot(np.arange(0, len(data['train_score'])), data['train_score'])
		#plt.plot(data['test_score'])
		#plt.legend(['Training', 'Validation'])
		#plt.show()
		#plot cdf
		error = [math.fabs(100*(x - y)) for x, y in zip(trainPredict, trainY)]
		plt.figure()
		plt.xlabel('Absolute error of PDR prediction over the validation data')
		plt.ylabel('CDF')
		plt.hist(error, normed=1, cumulative=True, label='CDF', histtype='step',  bins=1000)
		#plt.show()


		for i in range(3,4):
			mode = i
			if mode == 0:
				dataframe = read_csv(arglist.training_dir + 'exp2_KNN_pos_SNR.csv', usecols=[5*(arglist.look_ahead+1)],header=None, sep=',', engine='python')
			elif mode ==1:
				dataframe = read_csv(arglist.training_dir + 'exp2_KNN_pos_SNR.csv', usecols=[*range(0,5), 5*(arglist.look_ahead+1)],header=None, sep=',', engine='python')
			elif mode ==2:
				dataframe = read_csv(arglist.training_dir + 'exp2_KNN_pos_SNR.csv', usecols=[*range(0, (1+5*(arglist.look_ahead+1)))],header=None, sep=',', engine='python')
			elif mode == 3:
				dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[*range(0,(180+5*(arglist.look_ahead+1)))],header=None, sep=',', engine='python')

			datasetInput = dataframe.values
			datasetInput = datasetInput.astype('float32')

			# load the Output dataset
			if mode == 3:
				dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[180+5*(arglist.look_ahead+1)],header=None, engine='python')
			else:
				dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[1+5*(arglist.look_ahead+1)],header=None, engine='python')
			datasetOutput = dataframe.values
			datasetOutput = datasetOutput.astype('float32')
			# split into train and test sets
			train_size_input = int(len(datasetInput) * arglist.training_percentage)
			test_size_input = len(datasetInput) - train_size_input
			train_size_output = int(len(datasetOutput) * arglist.training_percentage)
			test_size_output = len(datasetOutput) - train_size_output
			trainX, testX = datasetInput[0:train_size_input,:], datasetInput[train_size_input:len(datasetInput),:]
			trainY, testY = datasetOutput[0:train_size_output], datasetOutput[train_size_output:len(datasetOutput)]

			rmse_val = [] #to store rmse values for different k
			#for K in range(10):
		#		K = K+1
			K = 20
			model = neighbors.KNeighborsRegressor(n_neighbors = K)

			model.fit(trainX, trainY)  #fit the model
			pred=model.predict(testX) #make prediction on test set
			error = math.sqrt(mean_squared_error(testY,pred)) #calculate rmse
			abs_error = [math.fabs(100*(x - y)) for x, y in zip(testY, pred)]
			plt.hist(abs_error, normed=1, cumulative=True,  label='CDF', histtype='step', bins=1000)
			rmse_val.append(error) #store rmse values
		plt.legend(['LSTM','KNN'])
		plt.show()

if __name__ == '__main__':
	arglist = parse_args()
	train(arglist)
