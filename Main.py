import PullData
import PullData2
import Net
import FormatData
import numpy as np

PullData.PullData()
#data = FormatData.FormatedData(.75, 2000)

#trainingData = zip(np.load('Inputs.npy'),np.load('Outputs.npy'))
#testData = zip(np.load('TestInputs.npy'),np.load('TestOutputs.npy'))

#net = Net.Net([3033,50,50,505],['LReLU','LReLU','sig','sig'], perc_dropout = 0.0, cost_func = 'MSE')
#net.SGD(trainingData,10.0,1000000,30,testData)
#net.adadelta(trainingData, 0.9, 0.00001, 100000000, testData, 10, 100000000, True)

#net = Net.Net([3033,108,88,505],['LReLU','LReLU','sig','sig'], cost_func = 'MSE', load_data = True)
#net.evaluate(testData, True, False)

#threshold = 0.0

#for layer in xrange(2):
	#count = 0
	#for i in xrange(net.sizes[layer + 1]):
		#sums = np.sum(net.weights[layer], axis = 0)[i] + net.biases[layer][i]
		#if sums > threshold:
			#print sums
			#count += 1

	#print str(count) + "/" + str(net.sizes[layer + 1])