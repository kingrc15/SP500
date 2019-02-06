import numpy as np
import sys
import matplotlib.pyplot as plt

class ActFuncError(Exception):
	pass

class ActFuncNotFound(Exception):
	pass

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
    	y.shape = (len(y),1)
        return (a-y)

#Activation Functions
class TanH(object):

	@staticmethod
	def func(x):
		return np.tanh(x)

	@staticmethod
	def prime(x):
		return 1.0 - np.tanh(x)**2

class ReLU(object):

	@staticmethod
	def func(x):
		return x * (x > 0)

	@staticmethod
	def prime(x):
		return 1 * (x >= 0)

class sig(object):

	@staticmethod
	def func(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def prime(x):
		return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))


class Net():
	def __init__(self, sizes, act_function, cost_func = 'CrossEntropyCost'):
		self.best = np.inf
		self.sizes = sizes
		self.layers = len(sizes)
		self.weights  = [np.random.rand(y,x)/np.sqrt(x) for x, y in zip(sizes[:-1],sizes[1:])]
		self.biases = [np.random.rand(y,1) for y in sizes[1:]]
		self.cost_func = getattr(sys.modules[__name__], cost_func)
		if len(act_function) == self.layers:
			self.act_function = [getattr(sys.modules[__name__], x) for x in act_function]
		else:
			raise ActFuncError("{0} activation functions specified when {1} were needed".format(len(act_function),self.layers))
	
	def feedforward(self, activation):
		activation.shape = (len(activation),1)
		for b, w, f in zip(self.biases, self.weights, self.act_function):
			activation = f.func(w.dot(activation)+b)
		return activation

	def GD(self, data, LR, epochs, batch_size, test_data):
		self.LR = LR
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
					self.update_batch(batch, self.LR)
					print "Epoch {0}: Cost = {1}".format(ec, float(self.evaluate(test_data, plot)))
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
			z = w.dot(activation)+b
			zs.append(z)
			activation = f.func(z)
			activations.append(activation)
		d = self.cost_func.delta(zs[-1], activations[-1],y)
		#d.shape = (len(d),1)
		delta_b[-1] = d
		delta_w[-1] = d.dot(activations[-2].transpose())
		for l in xrange(2, self.layers):
			ap = self.act_function[-l].prime(zs[-l])
			d = np.dot(np.transpose(self.weights[-l+1]), d)* ap
			delta_b[-l] = d
			delta_w[-l] = d.dot(activations[-l-1].transpose())

		return delta_w, delta_b

	def evaluate(self, data, plot):
		cost = 0.0
		for x, y in data:
			a = self.feedforward(x)
			cost += self.cost_func.fn(a, y)/len(data)
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