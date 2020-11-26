# LSTM for international airline passengers problem with regression framing
import numpy
import pandas as pd
import csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
import numpy as np
# convert an array of values into a dataset matrix
import tensorflow as tf
print(tf.__version__)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(dataset[i + look_back, :])
	return numpy.array(dataX), numpy.array(dataY)


# fix random seed for reproducibility
numpy.random.seed(7)


# load the dataset
dataframe = pd.read_csv('data/monthly-network-data.csv', usecols=[0, 1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

groundtruth_data = dataset
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.7)
validation_size = int(len(dataset) * 0.08)
test_size = len(dataset) - train_size
train, validation, test = dataset[0:train_size,:], dataset[train_size:train_size+validation_size,:], dataset[train_size+validation_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
validationX, validationY = create_dataset(validation, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[2]))
validationX = numpy.reshape(validationX, (validationX.shape[0], 1, validationX.shape[2]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[2]))

print(trainX.shape[1], trainX.shape[2])
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.1))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, epochs=250, batch_size=20, validation_data=(validationX, validationY), verbose=1, shuffle=False)
plt.figure()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('figs/nwpred-loss.png')

# make predictions
trainPredict = model.predict(trainX)
validPredit = model.predict(validationX)
testPredict = model.predict(testX)

print(np.shape(trainPredict))
print(np.shape(testPredict))


print(mean_squared_error(trainY, trainPredict))
print(mean_squared_error(testY, testPredict))

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))





for i in range(0, 2):
	labels = ['Users', 'Downlink_PRB']
	plt.figure()
	tesz =len(testY)
	aa=[x for x in range(tesz)]
	plt.plot(aa, testY[:tesz,i], marker='.', label="actual")
	plt.plot(aa, testPredict[:tesz,i], 'r', label="prediction")
	plt.ylabel(labels[i], size=15)
	plt.xlabel('Time', size=15)
	plt.legend(fontsize=13)
	figname = 'figs/nw-pred-' + labels[i].lower() + '.png'
	plt.savefig(figname)


# invert predictions
itrainPredict = scaler.inverse_transform(trainPredict)
itrainY = scaler.inverse_transform(trainY)
itestPredict = scaler.inverse_transform(testPredict)
itestY = scaler.inverse_transform(testY)

groundtruth_test = groundtruth_data[train_size+validation_size:len(dataset),:]
result = np.column_stack((itestY, groundtruth_test[:-2]))


df_out = pd.DataFrame(data = result,
                  columns = ['Predicted_Users', 'Predicted_ Downlink_PRB', 'GT_Users', 'GT_Downlink_PRB'])

df_out.to_csv('results/nw-pred-results.csv', index=False)


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(itrainY, itrainPredict))
print('INVTF Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(itestY, itestPredict))
print('INVTF Test Score: %.2f RMSE' % (testScore))



for i in range(0, 2):
	labels = ['Users', 'Downlink_PRB']
	plt.figure()
	tesz =len(itestY)
	aa=[x for x in range(tesz)]
	plt.plot(aa, itestY[:tesz,i], marker='.', label="actual")
	plt.plot(aa, itestPredict[:tesz,i], 'r', label="prediction")
	plt.ylabel(labels[i], size=15)
	plt.xlabel('Time', size=15)
	plt.legend(fontsize=13)
	figname = 'figs/invtf-nw-pred-' + labels[i].lower() + '.png'
	plt.savefig(figname)




# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan


# print(len(trainPredict)+(look_back*2)+2+len(validPredit))
# print(len(dataset)-1)
# print(len(testPredict))


testPredictPlot[len(trainPredict)+(look_back*2)+ 2+ len(validPredit):len(dataset)-2, :] = testPredict
plt.figure()
plt.plot(scaler.inverse_transform(dataset))
plt.plot(testPredictPlot)
plt.savefig('figs/nw-shft-invtf.png')


