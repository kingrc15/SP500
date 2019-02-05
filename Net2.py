import numpy as np
import matplotlib.pyplot as plt

class ActFuncError(Exception):
	pass

class ActFuncNotFound(Exception):
	pass

class Net():
	def __init__(self, sizes, act_function):
		self.sizes = sizes
		self.layers = len(sizes)
		self.weights  = [np.random.rand(y,x)/np.sqrt(x) for x, y in zip(sizes[:-1],sizes[1:])]
		self.biases = [np.random.rand(y,1) for y in sizes[1:]]
		if len(act_function) == self.layers:
			self.act_function = act_function
		else:
			raise ActFuncError("{0} activation functions specified when {1} were needed".format(len(act_function),self.layers))
	
	def feedforward(self, activation):
		activation.shape = (len(activation),1)
		for b, w, f in zip(self.biases, self.weights, self.act_function):
			if f == 'sig':
				activation = sigmoid(w.dot(activation)+b)
			#elif f == 'ReLU':						one day
			else:
				ActFuncNotFound("Activation function {0} not found.".format(f))
		return activation

	def GD(self, data, LR, epochs, batch_size, test_data):
		batch = []
		ec = 1
		epochs += 1
		while ec < epochs:
			bs = 0
			while bs < xrange(len(data)-2):
				if bs == 15:
					break
				batch.append(data[bs])
				if bs % batch_size == 0:
					if ec % 1000 == 0:
						plot = True
					else:
						plot = False
					self.update_batch(batch, LR)
					print "Epoch {0}: Cost = {1}".format(ec, float(self.evaluate(data, plot)))
					ec += 1
					if ec == epochs: break

	def update_batch(self, batch, LR):
		sum_b_d = [np.zeros(b.shape) for b in self.biases]
		sum_w_d = [np.zeros(w.shape) for w in self.weights]

		for x, y in batch:
			delta_w, delta_b = self.back_propagation(x, y)
			new_w = [nw + dnw for nw, dnw in zip(sum_w_d, delta_w)]
			new_b = [nb + dnb for nb, dnb in zip(sum_b_d, delta_b)]
		self.weight = [w-(LR/len(batch))*nw for w, nw in zip(self.weights, new_w)]
		self.biases = [b-(LR/len(batch))*nb for b, nb in zip(self.biases, new_b)]

	def cost_derivative(self, output, expected):
		return (output-expected)

	def back_propagation(self,x, y):
		delta_b = [np.zeros(b.shape) for b in self.biases]
		delta_w = [np.zeros(w.shape) for w in self.weights]

		zs = []
		activation = x
		activation.shape = (len(activation),1)
		activations = [x]
		for b, w, f in zip(self.biases, self.weights, self.act_function):
			if f == 'sig':
				z = w.dot(activation)+b
				zs.append(z)
				activation = sigmoid(z)
				activations.append(activation)
		cd = self.cost_derivative(activation[-1],y)
		cd.shape = (len(cd),1)
		d = cd * sig_prime(zs[-1])
		delta_b[-1] = d
		delta_w[-1] = d.dot(activations[-2].transpose())
		for l in xrange(2, self.layers):
			sp = sig_prime(zs[-l])
			d = np.dot(np.transpose(self.weights[-l+1]), d)* sp
			delta_b[-l] = d
			delta_w[-l] = d.dot(activations[-l-1].transpose())

		return delta_w, delta_b

	def evaluate(self, data, plot):
		cost = 0
		ys = np.zeros([1,len(data)])
		outputs = np.zeros([1,len(data)])
		i = 0
		for x, y in data:
			y.shape = (len(y),1)
			output = self.feedforward(x)
			ys[i] = sum(y)
			outputs[i] = sum(output)
			cost += abs(sum(y - output))
		if plot:
			plt.clf()
			com_num = np.arange(0, 505)
			com_num.shape = (1,505)
			outputs.transpose()
			ys.transpose()
			#outputs = [1,[outputs[x] for x in xrange(len(outputs))] for y in xrange(len(outputs))]
			#ys = [1,[ys[x] for x in xrange(len(ys))] for y in xrange(len(y))]
			plt.plot(com_num, outputs, 'ro', com_num, ys, 'bs')
			plt.show()
		return cost

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sig_prime(x):
	return sigmoid(x)*(1-sigmoid(x))