import numpy as np
import sys
import matplotlib.pyplot as plt
import math

#np.set_printoptions(threshold=np.nan)

class ActFuncError(Exception):
	pass

class ActFuncNotFound(Exception):
	pass

class MSE(object):
	@staticmethod
	def fn(a, y):
		return np.sum((a - y)**2) / y.size

	@staticmethod
	def delta(z, a, y):
		y.shape = (len(y),1)
		return a - y

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
    	y.shape = (len(y),1)
        return (a-y)


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
		return x * (x > 0.0)

	@staticmethod
	def prime(x):
		return 1.0 * (x >= 0.0)

class LReLU(object):
 
	@staticmethod
	def func(x):
		for i in xrange(len(x)):
			if x[i] > 0:
				x[i] = x[i]
			elif x[i] <= 0:
				x[i] = x[i] * 0.01
		return x

	@staticmethod
	def prime(x):
		for i in xrange(len(x)):
			if x[i] > 0:
				x[i] = 1
			elif x[i] < 0:
				x[i] = 0.01
		return x

class sig(object):

	@staticmethod
	def func(x):
		return 1.0 / (1.0 + np.exp(-x))

	@staticmethod
	def prime(x):
		return (1.0 / (1.0 + np.exp(-x))) * (1.0 - (1.0 / (1.0 + np.exp(-x))))


def RMS(gs, const):
	return [np.sqrt(g + const) for g in gs]


class Net():
	def __init__(self, sizes, act_function, perc_dropout = 0.0, cost_func = 'CrossEntropyCost', load_data = 'None'):
		self.best = np.inf
		self.sizes = sizes
		self.layers = len(sizes)
		self.perc_dropout = perc_dropout
		self.node_strength = [np.zeros((y,1)) for y in self.sizes[1:]]
		if load_data == 'None':
			self.weights  = [np.random.rand(y,x)*np.sqrt(0.0000001/(float(x)+float(y))) for x, y in zip(sizes[:-1],sizes[1:])]
			self.biases = [np.random.rand(y,1) for y in sizes[1:]]
			self.updatedropout()
		else:
			self.weights = np.load('weights.npy')
			self.biases = np.load('biases.npy')
			print 'Weights and Biases loaded.'
		self.cost_func = getattr(sys.modules[__name__], cost_func)
		self.min_cost = 1000000.0
		if len(act_function) == self.layers:
			self.act_function = [getattr(sys.modules[__name__], x) for x in act_function]
		else:
			raise ActFuncError("{0} activation functions specified when {1} were needed".format(len(act_function),self.layers))
	
	def updatedropout(self):
		self.dropout = [(self.perc_dropout < np.random.rand(y,1)) for y in self.sizes[1:]]
		self.dropout[self.layers-2] = np.ones(self.dropout[self.layers-2].shape)
		self.dropout[0] = (0.1 < np.random.rand(self.sizes[1],1))

	def feedforward(self, activation):
		activation.shape = (len(activation),1)
		for b, w, f in zip(self.biases, self.weights, self.act_function[1:]):
			activation = f.func(w.dot(activation)+b)
		return activation

	def adadelta(self, data, decay_rate, const, epochs, test_data, batch_size, plot_int, save):
		time = 0
		previous_w_grad = [np.zeros(w.shape) for w in self.weights]
		previous_b_grad = [np.zeros(b.shape) for b in self.biases]
		previous_w_update = [np.zeros(w.shape) for w in self.weights]
		previous_b_update = [np.zeros(b.shape) for b in self.biases]
		ec = 0

		while ec < epochs:
			for x, y in data:
				delta_w, delta_b = self.back_propagation(x, y)
				self.updatedropout()
				w_grad_sqr = np.multiply(previous_w_grad,decay_rate) + np.power(np.multiply((1.0-decay_rate),delta_w),2)
				b_grad_sqr = np.multiply(previous_b_grad,decay_rate) + np.power(np.multiply((1.0-decay_rate),delta_b),2)
				w_delta = -np.divide(RMS(previous_w_update, const),RMS(w_grad_sqr,const))*delta_w
				b_delta = -np.divide(RMS(previous_b_update, const),RMS(b_grad_sqr,const))*delta_b
				w_update_sqr = np.multiply(previous_w_update,decay_rate) + np.power(np.multiply((1.0-decay_rate),w_delta),2)
				b_update_sqr = np.multiply(previous_b_update,decay_rate) + np.power(np.multiply((1.0-decay_rate),b_delta),2)

				#self.recordNS()
				self.weights = [w + nw for w, nw in zip(self.weights, w_delta)]
				self.biases = [b + nb for b, nb in zip(self.biases, b_delta)]
				previous_w_grad = w_grad_sqr
				previous_b_grad = b_grad_sqr
				previous_w_update = w_update_sqr
				previous_b_update = b_update_sqr
				ec = time/batch_size
				if time % batch_size == 0:
					if ec % plot_int == 0:
						print "Epoch {0}: Cost = {1}".format(ec, float(self.evaluate(test_data, True, save)))
					else:
						print "Epoch {0}: Cost = {1}".format(ec, float(self.evaluate(test_data, False, save)))
				if ec == epochs: 
					print "Epoch {0}: Cost = {1}".format(ec, float(self.evaluate(test_data, True, save)))
					break
				time += 1

	def SGD(self, data, LR, epochs, batch_size, test_data):
		if test_data: n_test = len(test_data)
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
					if ec % 25 == 0:
						plot = True
					else:
						plot = False
					self.update_batch(batch, self.LR)
					print "Epoch {0}: Cost = {1}".format(ec, float(self.evaluate(test_data, True, save)))
					ec += 1
					if ec == epochs: break

	def recordNS(self):
		#print self.layers
		for layer in xrange(self.layers-2):
			for i in xrange(self.sizes[layer+1]):
				sums = np.concatenate((sums,np.sum(self.weights[layer], axis = 0)[i] + self.biases[layer][i]))
			print sums 
			self.node_strength[layer] = sums

	def update_batch(self, batch, LR):
		sum_b_d = [np.zeros(b.shape) for b in self.biases]
		sum_w_d = [np.zeros(w.shape) for w in self.weights]

		for x, y in batch:
			delta_w, delta_b = self.back_propagation(x, y)
			new_w = [nw + dnw for nw, dnw in zip(sum_w_d, delta_w)]
			new_b = [nb + dnb for nb, dnb in zip(sum_b_d, delta_b)]
		self.weights = [w-(LR/len(batch))*nw for w, nw in zip(self.weights, new_w)]
		self.biases = [b-(LR/len(batch))*nb for b, nb in zip(self.biases, new_b)]

	def back_propagation(self,x, y):
		delta_b = [np.zeros(b.shape) for b in self.biases]
		delta_w = [np.zeros(w.shape) for w in self.weights]

		zs = []
		activation = x
		activation.shape = (len(activation),1)
		activations = [x]
		for b, w, drop, f in zip(self.biases, self.weights, self.dropout, self.act_function):
			z = np.multiply(w.dot(activation)+b,drop)
			zs.append(z)
			activation = np.multiply(f.func(z),drop)
			activations.append(activation)
		cost = self.cost_func.delta(zs[-1], activations[-1],y)
		af = self.act_function[-1].prime(zs[-1])
		d = cost * af
		delta_b[-1] = d
		delta_w[-1] = d.dot(activations[-2].transpose())
		for l in xrange(2, self.layers):
			z = zs[-l]
			ap = self.act_function[-l].prime(z)
			d = np.dot(np.transpose(self.weights[-l+1]), d)* ap
			delta_b[-l] = d
			delta_w[-l] = d.dot(activations[-l-1].transpose())

		return (delta_w, delta_b)

	def evaluate(self, data, plot, save):
		c = 0
		cost = 0.0
		ys = np.zeros([len(data),1])
		outputs = np.zeros([(len(data)),1])
		col_num = np.zeros([(len(data)),1])
		for x, y in data:
			a = self.feedforward(x)
			cost += self.cost_func.fn(a, y)
			outputs[c] = np.average(a)
			ys[c] = np.average(y)
			col_num[c] = c
			c += 1
		if cost < self.min_cost:
			self.min_cost = cost
			if save:
				w=np.array(self.weights)
				b=np.array(self.biases)
				np.save('weights',w)
				np.save('biases',b)
				print "saved"
		if plot:
			plt.clf()
			plt.plot(col_num, outputs, 'ro', col_num, ys, 'bs')
			plt.show()
		return cost