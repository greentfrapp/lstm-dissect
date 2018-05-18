import tensorflow as tf
import numpy as np

from LSTM import LSTM
import utils


sess = tf.Session()
lstm = LSTM(sess, 1)
sess.run(tf.global_variables_initializer())
weights = sess.run(lstm.cells[0].weights)
lstm.load_weights(weights)
loss, gradients = lstm.fit([[[0]], [[0]], [[0]]], [[[3]], [[3]], [[3]]])
print(loss)
for i in np.arange(1000):
	for j, gradient in enumerate(gradients):
		gradients[j] = utils.average_gradients(gradient)
	gradients = utils.average_gradients(gradients)
	weights = [weights[i] - 1e-3 * grad for i, grad in enumerate(gradients)]
	lstm.load_weights(weights)
	loss, gradients = lstm.fit([[[0]], [[0]], [[0]]], [[[3]], [[3]], [[3]]])
	if i % 100 == 0:
		print(loss)