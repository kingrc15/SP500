import PullData
import PullData2
import Net
import FormatData
import numpy as np

#PullData2.PullData()
#data = FormatData.FormatedData(.75, 2000)
net = Net.Net([3033,500,500,505],['TanH','TanH','TanH','sig'], cost_func = 'MSE')

trainingData = zip(np.load('Inputs.npy'),np.load('Outputs.npy'))
testData = zip(np.load('TestInputs.npy'),np.load('TestOutputs.npy'))
#net.SGD(trainingData,5000.0,1000000,30,testData)
net.adadelta(trainingData, 0.9, 0.000001, 10000, testData, 1000)


#SGD  3033,500,505  ~  1623.53575109    LR = 10  MSE  TanH TanH sig
#SGD  3033,400,505  ~  2750 	  LR = 5   MSE  TanH TanH sig