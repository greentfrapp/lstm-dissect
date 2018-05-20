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


