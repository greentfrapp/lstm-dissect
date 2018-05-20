import numpy as np


class CountingGame(object):
	
	def __init__(self):
		super(CountingGame, self).__init__()

	def generate(self, length, samples):
		x = []
		y = []
		for i in np.arange(samples):
			count = 0
			x_seq = []
			y_seq = []
			for j in np.arange(length):
				# Count number of 1's
				# Restart count at -1
				# step = np.random.choice([-1, 0, 1])
				step = np.random.choice([0, 1])
				x_seq.append([step])
				count += step
				y_step = np.zeros(length + 1)
				y_step[count] = 1
			x.append(x_seq)
			y.append(y_step)
		return x, y

class CountingGame2(object):

	def __init__(self):
		super(CountingGame2, self).__init__()

	def generate(self, length, samples):
		x = []
		y = []
		for i in np.arange(samples):
			count = 0
			x_seq = []
			y_seq = []
			for j in np.arange(length):
				# Count number of 1's
				# Restart count at -1
				step = np.random.choice([-1, 0, 1], p=[0.1, 0.45, 0.45])
				x_seq.append([step])
				if step == -1:
					count = 0
				else:
					count += step
			y_step = np.zeros(length + 1)
			y_step[count] = 1
			x.append(x_seq)
			y.append(y_step)
		for i in np.arange(50):
			x.append(np.expand_dims(np.ones(length) * -1, axis=1))
			y.append(np.eye(length + 1)[0])
			x.append(np.expand_dims(np.zeros(length), axis=1))
			y.append(np.eye(length + 1)[0])
		return x, y
