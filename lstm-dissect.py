import tensorflow as tf
import numpy as np
import pickle


from LSTMSingle import LSTM
from CountingGame import CountingGame
import utils


epochs = 10000
batchsize = 5
shuffle_x = np.random.RandomState(42)
shuffle_y = np.random.RandomState(42)

task = CountingGame()
x, y = task.generate(length=10, samples=600)
test_x, test_y = task.generate(length=10, samples=1)

sess = tf.Session()
lstm = LSTM(sess, 128, 10)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=3)

# saver.restore(sess, "model/modelsingle")
"""
test = [[1], [1], [1], [1], [1]]
predictions = lstm.test(test)
for n, entry in enumerate(predictions):
	max_pred.append([np.argmax(entry)])
	confid.append(entry[0, np.argmax(entry)])
print(test)
print(predictions)
#"""

#"""
lstm_weights = sess.run(lstm.cells[0].lstm_weights)
lstm.load_weights(lstm_weights)

initial_lr = 1e0
lr_delta = (1e-3 - 1e0) / 499

initial_lr = 1e-3
lr_delta = 0

n_iters = len(x) / batchsize
for i in np.arange(epochs):
	lr = max(initial_lr + lr_delta * i, 1e-3)
	# lr = 1e-3
	shuffle_x.shuffle(x)
	shuffle_y.shuffle(y)
	for j in np.arange(n_iters):
		start = int(j * batchsize)
		end = int(start + batchsize)
		loss, lstm_gradients = lstm.fit(x[start:end], y[start:end])
		lstm_gradients = utils.average_gradients(lstm_gradients)
		lstm_weights = [lstm_weights[i] - lr * grad for i, grad in enumerate(lstm_gradients)]
		lstm.load_weights(lstm_weights)
	if i % 5 == 0:
		print("\nEpoch #{} Loss: {}".format(i, loss))
		test_x[0] = [[1], [1], [1], [-1], [1], [1], [-1], [1], [1], [1]]
		test_x[0] = [[1], [1], [1], [1], [1]]
		options = [
			[[1], [0], [0], [0], [1], [1], [0], [0], [0], [1]],
			[[0], [1], [0], [1], [0], [0], [1], [0], [1], [0]],
			[[1], [0], [1], [0], [0], [1], [0], [1], [0], [0]],
			[[1], [1], [0], [0], [0], [1], [1], [0], [0], [0]],
			[[1], [0], [0], [1], [0], [1], [0], [0], [1], [0]],
		]
		test_x[0] = options[np.random.choice(5)]
		corr_pred = [[1], [2], [3], [0], [1], [2], [0], [1], [2], [3]]
		corr_pred = [[1], [2], [3], [4], [5]]
		print(test_x[0])
		predictions = lstm.test(test_x[0])
		print(np.argmax(predictions))
		# max_pred = []
		# confid = []
		# corr_confid = []
		# for n, entry in enumerate(predictions):
		# 	max_pred.append([np.argmax(entry)])
		# 	confid.append(entry[0, np.argmax(entry)])
		# 	corr_confid.append(entry[0, corr_pred[n][0]])
		# print(max_pred)
		# print(confid)
		# print(corr_pred)
		# print(corr_confid)
		saver.save(sess, "model/modelsingle", global_step=i)

#"""
