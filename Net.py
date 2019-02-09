import numpy as np
import sys
import matplotlib.pyplot as plt

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
		#print x
		#print np.tanh(x)
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


def RMS(gs, const):
	return [np.sqrt(g + const) for g in gs]


class Net():
	def __init__(self, sizes, act_function, cost_func = 'CrossEntropyCost', load_data = 'None'):
		self.best = np.inf
		self.sizes = sizes
		self.layers = len(sizes)
		if load_data == 'None':
			self.weights  = [np.random.rand(y,x)/np.sqrt(x) for x, y in zip(sizes[:-1],sizes[1:])]
			self.biases = [np.random.rand(y,1) for y in sizes[1:]]
		else:
			self.weights = np.load('weights.npy')
			self.biases = np.load('biases.npy')
		self.cost_func = getattr(sys.modules[__name__], cost_func)
		self.min_cost = 1000000.0
		if len(act_function) == self.layers:
			self.act_function = [getattr(sys.modules[__name__], x) for x in act_function]
		else:
			raise ActFuncError("{0} activation functions specified when {1} were needed".format(len(act_function),self.layers))
	
	def feedforward(self, activation):
		activation.shape = (len(activation),1)
		for b, w, f in zip(self.biases, self.weights, self.act_function):
			activation = f.func(w.dot(activation)+b)
		return activation

	def adadelta(self, data, decay_rate, const, epochs, test_data, batch_size):
		time = 0
		previous_w_grad = [np.zeros(w.shape) for w in self.weights]
		previous_b_grad = [np.zeros(b.shape) for b in self.biases]
		previous_w_update = [np.zeros(w.shape) for w in self.weights]
		previous_b_update = [np.zeros(b.shape) for b in self.biases]
		ec = 0

		while ec < epochs:
			for x, y in data:
				delta_w, delta_b = self.back_propagation(x, y)
				w_grad_sqr = np.multiply(previous_w_grad,decay_rate) + np.power(np.multiply((1.0-decay_rate),delta_w),2)
				b_grad_sqr = np.multiply(previous_b_grad,decay_rate) + np.power(np.multiply((1.0-decay_rate),delta_b),2)
				w_delta = -np.divide(RMS(previous_w_update, const),RMS(w_grad_sqr,const))*delta_w
				b_delta = -np.divide(RMS(previous_b_update, const),RMS(b_grad_sqr,const))*delta_b
				w_update_sqr = np.multiply(previous_w_update,decay_rate) + np.power(np.multiply((1.0-decay_rate),w_delta),2)
				b_update_sqr = np.multiply(previous_b_update,decay_rate) + np.power(np.multiply((1.0-decay_rate),b_delta),2)

				self.weight = [w + nw for w, nw in zip(self.weights, w_delta)]
				self.biases = [b + nb for b, nb in zip(self.biases, b_delta)]
				previous_w_grad = w_grad_sqr
				previous_b_grad = b_grad_sqr
				previous_w_update = w_update_sqr
				previous_b_update = b_update_sqr
				ec = time/batch_size
				if time % batch_size == 0:
					if time / batch_size % batch_size == 0:
						print "Epoch {0}: Cost = {1}".format(ec, float(self.evaluate(test_data, True)))
					else:
						print "Epoch {0}: Cost = {1}".format(ec, float(self.evaluate(test_data, False)))
				if ec == epochs: 
					print "Epoch {0}: Cost = {1}".format(ec, float(self.evaluate(test_data, True)))
					break
				time += 1

	def SGD(self, data, LR, epochs, batch_size, test_data):
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
					if ec % 100 == 0:
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
		d = self.cost_func.delta(zs[-1], activations[-1],y) * self.act_function[-1].prime(zs[-1])
		delta_b[-1] = d
		delta_w[-1] = d.dot(activations[-2].transpose())
		for l in xrange(2, self.layers):
			ap = self.act_function[-l].prime(zs[-l])
			d = np.dot(np.transpose(self.weights[-l+1]), d)* ap
			delta_b[-l] = d
			delta_w[-l] = d.dot(activations[-l-1].transpose())

		return delta_w, delta_b

	def evaluate(self, data, plot):
		c = 0
		comp_num = 10
		cost = 0.0
		ys = np.zeros([len(data),1])
		outputs = np.zeros([(len(data)),1])
		col_num = np.zeros([(len(data)),1])
		for x, y in data:
			a = self.feedforward(x)
			cost += self.cost_func.fn(a, y)
			outputs[c] = a[comp_num]
			ys[c] = y[comp_num]
			col_num[c] = c
			c += 1
		if cost < self.min_cost:
			self.min_cost = cost
			w=np.array(self.weights)
			b=np.array(self.biases)
			np.save('weights',w)
			np.save('biases',b)
			#np.savetxt('weights',w,delimiter = ',')
			#np.savetxt('biases',b,delimiter = ',')
		if plot:
			plt.clf()
			plt.plot(col_num, outputs, 'ro', col_num, ys, 'bs')
			plt.show()
		return cost