import numpy as np


class CountingGame(object):
	
	def __init__(self):
		super(CountingGame, self).__init__()

	def generate(self, length):
		x = []
		y = []
		count = 0
		for i in np.arange(length):
			# Count number of 1's
			# Restart count at -1
			step = np.random.choice([-1, 0, 1])
			x.append(step)
			if step == -1:
				count = 0
			else:
				count += 1
			y.append(count)
		return x, y


