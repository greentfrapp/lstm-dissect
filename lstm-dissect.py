from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import tensorflow as tf
import numpy as np
import pickle
from absl import flags
from absl import app

from LSTMSingle import LSTM, LSTMStep
from RNN import RNN, RNNStep
from CountingGame import CountingGame, CountingGame2
import utils


FLAGS = flags.FLAGS

# General
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_bool("step", False, "Step")

flags.DEFINE_string("type", "lstm", "Type of RNN (lstm or rnn)")
flags.DEFINE_integer("hidden", 128, "Number of hidden nodes")

# Train
flags.DEFINE_integer("epochs", 1000, "Number of training epochs")
flags.DEFINE_integer("samples", 600, "Number of training samples from MNIST")
flags.DEFINE_integer("batchsize", 5, "Training batchsize")
flags.DEFINE_integer("seqlen", 10, "Sequence length for training")

# Test
flags.DEFINE_list("seq", None, "Sequence for testing")
flags.DEFINE_integer("val", 0, "Value for testing")


def train(option="lstm"):
	epochs = FLAGS.epochs
	batchsize = FLAGS.batchsize
	shuffle_x = np.random.RandomState(42)
	shuffle_y = np.random.RandomState(42)

	task = CountingGame2()
	x, y = task.generate(length=FLAGS.seqlen, samples=FLAGS.samples)
	test_x, test_y = task.generate(length=FLAGS.seqlen, samples=1)

	sess = tf.Session()
	if option == "lstm":
		lstm = LSTM(sess, FLAGS.hidden, FLAGS.seqlen)
	elif option == "rnn":
		lstm = RNN(sess, FLAGS.hidden, FLAGS.seqlen)

	sess.run(tf.global_variables_initializer())
	
	lstm_weights = sess.run(lstm.cells[0].lstm_weights)
	lstm.load_weights(lstm_weights)

	# initial_lr = 1e0
	# lr_delta = (1e-3 - 1e0) / 499

	initial_lr = 1e-3
	lr_delta = 0

	n_iters = len(x) / batchsize
	for i in np.arange(epochs):
		lr = max(initial_lr + lr_delta * i, 1e-3)
		shuffle_x.shuffle(x)
		shuffle_y.shuffle(y)
		for j in np.arange(n_iters):
			start = int(j * batchsize)
			end = int(start + batchsize)
			loss, lstm_gradients = lstm.fit(x[start:end], y[start:end])
			lstm_gradients = utils.average_gradients(lstm_gradients)
			lstm_weights = [lstm_weights[i] - lr * grad for i, grad in enumerate(lstm_gradients)]
			dense_weights = sess.run(lstm.dense_weights)
			lstm.load_weights(lstm_weights)
		if i % 5 == 0:
			print("\nEpoch #{} Loss: {}".format(i, loss))
			test_x[0] = [[1], [1], [1], [-1], [1], [1], [-1], [1], [1], [1]]
			test_x[0] = [[1], [1], [1], [1], [1]]
			options = [
				[[1], [0], [0], [-1], [1], [1], [0], [0], [0], [1]],
				[[0], [1], [-1], [1], [0], [0], [1], [0], [1], [0]],
				[[1], [0], [1], [0], [-1], [1], [0], [1], [1], [1]],
				[[1], [1], [-1], [0], [0], [1], [1], [0], [1], [1]],
				[[1], [-1], [1], [1], [0], [1], [0], [0], [1], [0]],
			]
			test_x[0] = options[np.random.choice(5)]
			corr_pred = [[1], [2], [3], [0], [1], [2], [0], [1], [2], [3]]
			corr_pred = [[1], [2], [3], [4], [5]]
			print(test_x[0])
			predictions = lstm.test(test_x[0])
			print(np.argmax(predictions))
			with open("model/rnn_lstm_weights_2_small.pkl", 'wb') as file:
				pickle.dump(lstm_weights, file)
			with open("model/rnn_dense_weights_2_small.pkl", 'wb') as file:
				pickle.dump(dense_weights, file)

def test(option="lstm"):
	if FLAGS.seq is None:
		ones = np.random.choice(np.arange(FLAGS.seqlen), FLAGS.val, replace=False)
		seq = np.zeros(FLAGS.seqlen)
		seq[ones] = 1
	else:
		seq = np.array(FLAGS.seq).astype(np.float32)
	seq = np.expand_dims(seq, axis=1)
	sess = tf.Session()
	if option == "lstm":
		lstm = LSTM(sess, FLAGS.hidden, FLAGS.seqlen)
	elif option == "rnn":
		lstm = RNN(sess, FLAGS.hidden, FLAGS.seqlen)
	sess.run(tf.global_variables_initializer())
	with open("model/rnn_lstm_weights_2_small.pkl", 'rb') as file:
		lstm_weights = pickle.load(file)
	with open("model/rnn_dense_weights_2_small.pkl", 'rb') as file:
		dense_weights = pickle.load(file)
	lstm.load_weights(lstm_weights, dense_weights)
	print(seq.reshape(-1))
	predictions = lstm.test(seq)
	print(np.argmax(predictions))

def step(option="lstm"):
	sess = tf.Session()
	if option == "lstm":
		lstm = LSTMStep(sess, FLAGS.hidden)
	elif option == "rnn":
		lstm = RNNStep(sess, FLAGS.hidden)
	sess.run(tf.global_variables_initializer())
	with open("model/rnn_lstm_weights_2_small.pkl", 'rb') as file:
		lstm_weights = pickle.load(file)
	with open("model/rnn_dense_weights_2_small.pkl", 'rb') as file:
		dense_weights = pickle.load(file)
	lstm.load_weights(lstm_weights, dense_weights)

	cell_state = hidden_state = np.zeros((1, FLAGS.hidden))
	while True:
		step = raw_input("Next Step: ")
		if step == 'q':
			quit()
		step = float(step)
		if option == "lstm":
			predictions, cell_state, hidden_state = lstm.step(step, cell_state, hidden_state)
			print(np.argmax(predictions))
			print(cell_state)
		elif option == "rnn":
			predictions, hidden_state = lstm.step(step, hidden_state)
			print(np.argmax(predictions))
			print(hidden_state)


def main(argv):
	if FLAGS.train:
		train(FLAGS.type)
	elif FLAGS.test:
		test(FLAGS.type)
	elif FLAGS.step:
		step(FLAGS.type)
	

if __name__ == "__main__":
	app.run(main)