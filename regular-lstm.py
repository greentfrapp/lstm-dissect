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
test_x, test_y = task.generate(length=5, samples=1)

sess = tf.Session()

inputs = tf.placeholder(
	shape=[None, 5, 1],
	dtype=tf.float32,
	name="inputs",
)
labels = tf.placeholder(
	shape=[None, 32],
	dtype=tf.float32,
	name="labels",
)

_enc_cell = tf.contrib.rnn.LSTMCell(32)
transformed_inputs = [tf.squeeze(t, [1]) for t in tf.split(inputs, 5, 1)]
batch_enc_output, batch_enc_state = tf.contrib.rnn.static_rnn(_enc_cell, transformed_inputs, dtype=tf.float32)
batch_enc_output = tf.stack(batch_enc_output)
batch_enc_output = tf.transpose(batch_enc_output, [1, 0, 2])
index = tf.range(0, batchsize) * 5 + 5 - 1
output_op = tf.gather(tf.reshape(batch_enc_output, [-1, 32]), index)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_op, labels=labels))
optimize_op = tf.train.AdamOptimizer().minimize(loss_op)
predictions_op = tf.nn.sigmoid(output_op)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

n_iters = len(x) / batchsize
for i in np.arange(epochs):
	shuffle_x.shuffle(x)
	shuffle_y.shuffle(y)
	for j in np.arange(n_iters):
		start = int(j * batchsize)
		end = int(start + batchsize)
		loss, _ = sess.run([loss_op, optimize_op], feed_dict={inputs: x[start:end], labels: np.array(y)[start:end, -1, :]})
	if i % 50 == 0:
		print("Epoch #{} Loss: {}".format(i, loss))
		print(test_x[0])
		predictions = sess.run(predictions_op, feed_dict={inputs: [test_x[0]]})
		max_pred = []
		for entry in predictions:
			max_pred.append(np.argmax(entry))
		print(max_pred)
saver.save(sess, "model/regular_model")