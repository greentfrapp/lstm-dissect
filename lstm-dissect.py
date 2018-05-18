import tensorflow as tf
import numpy as np

from LSTM import LSTM
from CountingGame import CountingGame
import utils


epochs = 1000
batchsize = 10
shuffle_x = np.random.RandomState(42)
shuffle_y = np.random.RandomState(42)

task = CountingGame()
x, y = task.generate(length=5, samples=1000)

sess = tf.Session()
lstm = LSTM(sess, 32, 5)
sess.run(tf.global_variables_initializer())
weights = sess.run(lstm.cells[0].weights)
lstm.load_weights(weights)

n_iters = len(x) / batchsize
for i in np.arange(epochs):
	shuffle_x.shuffle(x)
	shuffle_y.shuffle(y)
	for j in np.arange(n_iters):
		start = int(j * batchsize)
		end = int(start + batchsize)
		loss, gradients = lstm.fit(x[start:end], y[start:end])
		for j, gradient in enumerate(gradients):
			gradients[j] = utils.average_gradients(gradient)
		gradients = utils.average_gradients(gradients)
		weights = [weights[i] - 1e-3 * grad for i, grad in enumerate(gradients)]
		lstm.load_weights(weights)
	if i % 50 == 0:
		print("Epoch #{} Loss: {}".format(i, loss))