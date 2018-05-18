import numpy as np

def average_gradients(gradients):
	output = []
	for variables in zip(*gradients):
		output.append(np.mean(variables, axis=0))
	return output