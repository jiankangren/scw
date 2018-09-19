import argparse
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import os
import os.path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error


def parse_args():
	parser = argparse.ArgumentParser("PDR prediction experiment using LSTM network")
	# Environment
	parser.add_argument("--exp-name", type=str, default=None, help="name of the training experiment")
	parser.add_argument("--batch-size", type=int, default=20, help="number of sequences to optimize at the same time")
	parser.add_argument("--look-back", type=int, default=200, help="window size")
	# Training
	parser.add_argument("--training-input", type=str, default=None, help="name of the training input data file")
	parser.add_argument("--training-output", type=str, default=None, help="name of the training output data file")
	parser.add_argument("--training_dir", type=str, default="./dataset/", help="directory in which training data exists")
	parser.add_argument("--epoch-count", type=int, default=10, help="number of epochs used for training")
	parser.add_argument("--save-dir", type=str, default="./model/", help="directory in which training state and model should be saved")
	parser.add_argument("--save-rate", type=int, default=1, help="save model once every time this many epoch are completed")
	parser.add_argument("--load-dir", type=str, default="./model/", help="directory in which training state and model are loaded")
	# Evaluation
	parser.add_argument("--restore", action="store_true", default=False)
	parser.add_argument("--plots-dir", type=str, default="./figs/", help="directory where plot data is saved")
	return parser.parse_args()

# convert an array of values into a dataset matrix
def create_dataset(datasetInput, datasetOutput, look_back=1):
	dataX, dataY = [], []
	for i in range(len(datasetInput)-look_back):
		a = datasetInput[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(datasetOutput[i, 0])
	return numpy.array(dataX), numpy.array(dataY)
# trim the training and testing datasets to have entries multiple of batch_size
def trim_to_batch_size(dataset, batch_size):
	return dataset[0:len(dataset)-(len(dataset)%batch_size)]

def train(arglist):
	# fix random seed for reproducibility
	numpy.random.seed(7)
	# load the Input dataset
	dataframe = read_csv(arglist.training_dir + arglist.training_input, header=None, sep=',', engine='python')
	datasetInput = dataframe.values
	datasetInput = datasetInput.astype('float32')

	# load the Output dataset
	dataframe = read_csv(arglist.training_dir + arglist.training_output, usecols=[0],header=None, engine='python')
	datasetOutput = dataframe.values
	datasetOutput = datasetOutput.astype('float32')

	look_back = arglist.look_back
	batch_size = arglist.batch_size
	datasetInput, datasetOutput = create_dataset(datasetInput, datasetOutput, look_back)

	# split into train and test sets
	train_size_input = int(len(datasetInput) * 0.8)
	test_size_input = len(datasetInput) - train_size_input
	train_size_output = int(len(datasetOutput) * 0.8)
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
	trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 180))
	testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 180))

	# check if there a saved model exists
	modelLoadFilePath= arglist.load_dir + arglist.exp_name + '_model.json'
	if not os.path.isfile(modelLoadFilePath):
		# create and fit the LSTM network
		model = Sequential()
		#model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
		model.add(CuDNNLSTM(64, batch_input_shape=(batch_size, look_back, 180)))
		model.add(Dense(1))
		print("Created model")
	else:
		# load json and create model
		json_file = open(modelLoadFilePath, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		# load weights into new model
		model.load_weights(arglist.load_dir + arglist.exp_name + '_model.h5')
		print("Loaded model from disk")

	model.compile(loss='mean_squared_error', optimizer='adam')

	for i in range(arglist.epoch_count):
		model.fit(trainX, trainY, epochs=arglist.save_rate, batch_size=arglist.batch_size, verbose=2)
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
	# make predictions
	trainPredict = model.predict(trainX, batch_size=batch_size)
	# uncomment to use stateful LSTMS
	#model.reset_states()
	testPredict = model.predict(testX, batch_size=batch_size)
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	# plot results
	## plot prediction vs groundtruth of training data
	plt.figure()
	plt.plot(trainY)
	plt.plot(trainPredict)
	plt.show()
	## plot prediction vs groundtruth of testing data
	plt.figure()
	plt.plot(testY)
	plt.plot(testPredict)
	plt.show()

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
