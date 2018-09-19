# Stacked LSTM for international airline passengers problem with memory
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
# fix random seed for reproducibility
numpy.random.seed(7)
# load the Input dataset
dataframe = read_csv('test2FeaturesCSI-all.csv', header=None, sep=',', engine='python')
datasetInput = dataframe.values
datasetInput = datasetInput.astype('float32')

# load the Output dataset
dataframe = read_csv('test2LabelPDRtw1.csv', usecols=[0],header=None, engine='python')
datasetOutput = dataframe.values
datasetOutput = datasetOutput.astype('float32')

look_back = 200
batch_size = 20
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
modelFilePath='./model.json'
if not os.path.isfile(modelFilePath):
	# create and fit the LSTM network
	model = Sequential()
	#model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
	model.add(CuDNNLSTM(64, batch_input_shape=(batch_size, look_back, 180)))
	model.add(Dense(1))
	print("Created model")
else:
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("model.h5")
	print("Loaded model from disk")

model.compile(loss='mean_squared_error', optimizer='adam')

for i in range(10):
	model.fit(trainX, trainY, epochs=1, batch_size=20, verbose=2)
#   uncomment to use stateful LSTMS
#	model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2, shuffle=False)
#	model.reset_states()
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
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
