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
flags.DEFINE_float("lr", 0.001, "Learning rate")

# Test
flags.DEFINE_list("seq", None, "Sequence for testing")
flags.DEFINE_integer("val", 0, "Value for testing")


def train(option="lstm", file_desc=""):
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

	n_iters = len(x) / batchsize
	for i in np.arange(epochs):
		shuffle_x.shuffle(x)
		shuffle_y.shuffle(y)
		for j in np.arange(n_iters):
			start = int(j * batchsize)
			end = int(start + batchsize)
			loss, lstm_gradients = lstm.fit(x[start:end], y[start:end])
			lstm_gradients = utils.average_gradients(lstm_gradients)
			lstm_weights = [lstm_weights[i] - FLAGS.lr * grad for i, grad in enumerate(lstm_gradients)]
			dense_weights = sess.run(lstm.dense_weights)
			lstm.load_weights(lstm_weights)
		if i % 5 == 0:
			print("\nEpoch #{} Loss: {}".format(i, loss))
			print(test_x[0])
			predictions = lstm.test(test_x[0])
			print(np.argmax(predictions))
			with open("model/{}_lstm.pkl".format(file_desc), 'wb') as file:
				pickle.dump(lstm_weights, file)
			with open("model/{}_dense.pkl".format(file_desc), 'wb') as file:
				pickle.dump(dense_weights, file)

def test(option="lstm", file_desc=""):
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
	print("\n\nLoading model/{}_lstm.pkl...".format(file_desc))
	with open("model/{}_lstm.pkl".format(file_desc), 'rb') as file:
		lstm_weights = pickle.load(file)
	print("\n\nLoading model/{}_dense.pkl...\n\n".format(file_desc))
	with open("model/{}_dense.pkl".format(file_desc), 'rb') as file:
		dense_weights = pickle.load(file)
	lstm.load_weights(lstm_weights, dense_weights)
	print(seq.reshape(-1))
	predictions = lstm.test(seq)
	print(np.argmax(predictions))

def step(option="lstm", file_desc=""):
	sess = tf.Session()
	if option == "lstm":
		lstm = LSTMStep(sess, FLAGS.hidden)
	elif option == "rnn":
		lstm = RNNStep(sess, FLAGS.hidden)
	sess.run(tf.global_variables_initializer())
	print("\n\nLoading model/{}_lstm.pkl...".format(file_desc))
	with open("model/{}_lstm.pkl".format(file_desc), 'rb') as file:
		lstm_weights = pickle.load(file)
	print("\n\nLoading model/{}_dense.pkl...\n\n".format(file_desc))
	with open("model/{}_dense.pkl".format(file_desc), 'rb') as file:
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
	desc = "_".join([FLAGS.type, "h{}".format(FLAGS.hidden)])
	if FLAGS.train:
		train(FLAGS.type, desc)
	elif FLAGS.test:
		test(FLAGS.type, desc)
	elif FLAGS.step:
		step(FLAGS.type, desc)
	

if __name__ == "__main__":
	app.run(main)