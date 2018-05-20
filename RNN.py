import tensorflow as tf
import numpy as np


class RNN(object):

	def __init__(self, sess, units, seqlen):
		super(RNN, self).__init__()
		self.units = units
		self.sess = sess
		self.seqlen = seqlen
		self.cell = RNNCell
		self.build_model()
		self.unroll()
		self.dense_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dense")
		self.dense_weights_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in self.dense_weights]
		dense_assigns = [tf.assign(v, p) for v, p in zip(self.dense_weights, self.dense_weights_placeholders)]
		self.dense_assign = tf.group(*dense_assigns)

	def build_model(self):

		# initial hidden state
		self.hidden_state = tf.placeholder(
			shape=[None, self.units],
			dtype=tf.float32,
			name="initial_hidden_state",
		)

	def unroll(self):
		self.cells = []
		cell = self.cell(
			sess=self.sess,
			units=self.units, 
			hidden_state=self.hidden_state,
			name='0',
		)
		hidden_state = cell.new_hidden_state
		self.cells.append(cell)
		for i in np.arange(self.seqlen - 1):
			cell = self.cell(
				sess=self.sess,
				units=self.units, 
				hidden_state=hidden_state,
				name=str(i + 1),
			)
			hidden_state = cell.new_hidden_state
			self.cells.append(cell)

		with tf.variable_scope("dense"):
			output_dense_1 = tf.layers.dense(
				inputs=hidden_state,
				units=self.units,
				activation=tf.nn.relu,
				kernel_initializer=tf.truncated_normal_initializer(.0,.01),
				name="output_dense_1",
			)

			self.output = tf.layers.dense(
				inputs=output_dense_1,
				units=11,
				activation=None,
				kernel_initializer=tf.truncated_normal_initializer(.0,.01),
				name="output",
			)

		self.prediction = tf.nn.softmax(self.output)

		# Loss Function
		self.labels = tf.placeholder(
			shape=[None, 11],
			dtype=tf.float32,
			name="main_labels"
		)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output))
		self.lstm_gradients = []
		for i, cell in enumerate(self.cells):
			self.lstm_gradients.append(tf.gradients(self.loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}/rnn".format(cell.name))))

		self.dense_optimize = tf.train.AdamOptimizer().minimize(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dense"))

	def load_weights(self, lstm_weights, dense_weights=None):
		for i, cell in enumerate(self.cells):
			cell.load_weights(lstm_weights)
		if dense_weights is not None:
			feed_dict = dict(zip(self.dense_weights_placeholders, dense_weights))
			self.sess.run(self.dense_assign, feed_dict=feed_dict)

	def fit(self, x, y):
		feed_dict = {}
		for i, cell in enumerate(self.cells):
			feed_dict[cell.input] = np.array(x)[:, i]
		feed_dict[self.hidden_state] = np.zeros((len(x), self.units))
		feed_dict[self.labels] = y
		loss, lstm_gradients, _ = self.sess.run([self.loss, self.lstm_gradients, self.dense_optimize], feed_dict)
		return loss, lstm_gradients

	def test(self, x, hidden_state=None):
		feed_dict = {}
		for i, cell in enumerate(self.cells):
			feed_dict[cell.input] = [x[i]]
		feed_dict[self.hidden_state] = hidden_state or np.zeros((1, self.units))
		return self.sess.run(self.prediction, feed_dict)

class RNNStep(object):

	def __init__(self, sess, units):
		super(RNNStep, self).__init__()
		self.units = units
		self.sess = sess
		self.build_model()
		self.dense_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dense")
		self.dense_weights_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in self.dense_weights]
		dense_assigns = [tf.assign(v, p) for v, p in zip(self.dense_weights, self.dense_weights_placeholders)]
		self.dense_assign = tf.group(*dense_assigns)

	def build_model(self):

		# initial hidden state
		self.hidden_state = tf.placeholder(
			shape=[None, self.units],
			dtype=tf.float32,
			name="initial_hidden_state",
		)

		self.cell = RNNCell(
			sess=self.sess,
			units=self.units,
			hidden_state=self.hidden_state,
			name="cell",
		)

		with tf.variable_scope("dense"):
			output_dense_1 = tf.layers.dense(
				inputs=self.cell.new_hidden_state,
				units=self.units,
				activation=tf.nn.relu,
				kernel_initializer=tf.truncated_normal_initializer(.0,.01),
				name="output_dense_1",
			)

			self.output = tf.layers.dense(
				inputs=output_dense_1,
				units=11,
				activation=None,
				kernel_initializer=tf.truncated_normal_initializer(.0,.01),
				name="output",
			)

		self.prediction = tf.nn.softmax(self.output)

	def load_weights(self, lstm_weights, dense_weights):
		self.cell.load_weights(lstm_weights)
		feed_dict = dict(zip(self.dense_weights_placeholders, dense_weights))
		self.sess.run(self.dense_assign, feed_dict=feed_dict)

	def step(self, x, hidden_state=None):
		feed_dict = {}
		feed_dict[self.cell.input] = [[x]]
		if hidden_state is None:
			feed_dict[self.hidden_state] = np.zeros((1, self.units))
		else:
			feed_dict[self.hidden_state] = hidden_state
		return self.sess.run([self.prediction, self.cell.new_hidden_state], feed_dict)
		

class RNNCell(object):

	def __init__(self, sess, units, hidden_state, name):
		super(RNNCell, self).__init__()
		self.units = units
		self.hidden_state = hidden_state
		self.name = name
		self.sess = sess
		with tf.variable_scope(self.name):
			self.build_model()
			self.lstm_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}/rnn".format(self.name))
			self.lstm_weights_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in self.lstm_weights]
			lstm_assigns = [tf.assign(v, p) for v, p in zip(self.lstm_weights, self.lstm_weights_placeholders)]
			self.lstm_assign = tf.group(*lstm_assigns)

	def build_model(self):

		# input (per step)
		self.input = tf.placeholder(
			shape=[None, 1],
			dtype=tf.float32,
			name="{}_input".format(self.name),
		)

		with tf.variable_scope("rnn"):

			# new hidden state
			self.new_hidden_state = tf.layers.dense(
				inputs=tf.concat([self.hidden_state, self.input], axis=1),
				units=self.units,
				activation=tf.nn.relu,
				kernel_initializer=tf.truncated_normal_initializer(.0,.01),
				name="hidden_state",
			)

	def load_weights(self, lstm_weights):
		feed_dict = dict(zip(self.lstm_weights_placeholders, lstm_weights))
		self.sess.run(self.lstm_assign, feed_dict=feed_dict)








