import argparse
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import os
import os.path
import json
from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.layers import CuDNNLSTM
#from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
import pandas as pd




def parse_args():
	parser = argparse.ArgumentParser("PDR prediction experiment using LSTM network")
	# Environment
	parser.add_argument("--exp-name", type=str, default=None, help="name of the training experiment")
	parser.add_argument("--look-ahead", type=int, default=200, help="look ahead window size used for PDR calculations")
	parser.add_argument("--mode",type=int, default=0, help="0: use current pos only. 1: use current posand vel. 2: use current pos, vel, CSI. 3: use current pos, vel, CSI, future pos and vel")
	# Training
	parser.add_argument("--training-input", type=str, default=None, help="name of the training input data file")
	parser.add_argument("--training_dir", type=str, default="../dataset/", help="directory in which training data exists")
	parser.add_argument("--save-dir", type=str, default="../model/", help="directory in which training state and model should be saved")
	parser.add_argument("--load-dir", type=str, default="../model/", help="directory in which training state and model are loaded")
	parser.add_argument("--training-percentage", type=float, default=0.8, help="Percentage of the data to be used for training")
	# Evaluation
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

	if arglist.mode ==0:
		dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[5*(arglist.look_ahead+1)],header=None, sep=',', engine='python')
	elif arglist.mode ==1:
		dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[*range(0,5), 5*(arglist.look_ahead+1)],header=None, sep=',', engine='python')
	elif arglist.mode ==2:
		dataframe = read_csv(arglist.training_dir + arglist.training_input, usecols=[*range(0, (1+5*(arglist.look_ahead+1)))],header=None, sep=',', engine='python')
	datasetInput = dataframe.values
	datasetInput = datasetInput.astype('float32')

	# load the Output dataset
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
	K = 10
	model = neighbors.KNeighborsRegressor(n_neighbors = K)

	model.fit(trainX, trainY)  #fit the model
	pred=model.predict(testX) #make prediction on test set
	error = math.sqrt(mean_squared_error(testY,pred)) #calculate rmse
	abs_error = [math.fabs(x - y) for x, y in zip(testY, pred)]
	plt.hist(abs_error, normed=1, cumulative=True,  label='CDF', histtype='step', color='k', bins=1000)
	plt.show()
	rmse_val.append(error) #store rmse values
	print('RMSE value for k= ' , K , 'is:', error)
	#plotting the rmse values against k values
	#curve = pd.DataFrame(rmse_val) #elbow curve
	#curve.plot()


if __name__ == '__main__':
	arglist = parse_args()
	train(arglist)
