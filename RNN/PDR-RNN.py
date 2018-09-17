# Stacked LSTM for international airline passengers problem with memory
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(datasetInput, datasetOutput, look_back=1):
	dataX, dataY = [], []
	for i in range(len(datasetInput)-look_back-1):
		a = datasetInput[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(datasetOutput[i, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the Input dataset
dataframe = read_csv('test2FeaturesCSI-all.csv', engine='python')
datasetInput = dataframe.values
datasetInput = datasetInput.astype('float32')

# load the Output dataset
dataframe = read_csv('test2LabelPDRtw1.csv', usecols=[0], engine='python')
datasetOutput = dataframe.values
datasetOutput = datasetInput.astype('float32')

# split into train and test sets
train_size_input = int(len(datasetInput) * 0.8)
test_size_input = len(datasetInput) - train_size_input
train_size_output = int(len(datasetOutput) * 0.8)
test_size_output = len(datasetOutput) - train_size_output
trainInput, testInput = datasetInput[0:train_size_input,:], datasetInput[train_size_input:len(datasetInput),:]
trainOutput, testOutput = datasetInput[0:train_size_output,:], datasetInput[train_size_output:len(datasetInput),:]
# reshape into X=t and Y=t+1
look_back = 200
trainX, trainY = create_dataset(trainInput, trainOutput, look_back)
testX, testY = create_dataset(testInput, testOutput, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
batch_size = 1
model = Sequential()
#model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(100):
	model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()