import tensorflow as tf
import numpy as np


class LSTM(object):

	def __init__(self, sess, units, seqlen):
		super(LSTM, self).__init__()
		self.units = units
		self.sess = sess
		self.seqlen = seqlen
		self.cell = LSTMCell
		self.build_model()
		self.unroll()

	def build_model(self):

		# initial cell state
		self.cell_state = tf.placeholder(
			shape=[None, self.units],
			dtype=tf.float32,
			name="initial_cell_state",
		)

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
			cell_state=self.cell_state, 
			hidden_state=self.hidden_state,
			name='0',
		)
		cell_state = cell.new_cell_state
		hidden_state = cell.new_hidden_state
		self.cells.append(cell)
		for i in np.arange(self.seqlen - 1):
			cell = self.cell(
				sess=self.sess,
				units=self.units, 
				cell_state=cell_state, 
				hidden_state=hidden_state,
				name=str(i + 1),
			)
			cell_state = cell.new_cell_state
			hidden_state = cell.new_hidden_state
			self.cells.append(cell)

		# Loss Function
		self.labels = tf.placeholder(
			shape=[None, self.units],
			dtype=tf.float32,
			name="main_labels"
		)
		self.loss = tf.losses.mean_squared_error(self.labels, self.cells[-1].new_hidden_state)
		self.gradients = []
		for i, cell in enumerate(self.cells):
			self.gradients.append([])
			for j, cell2 in enumerate(self.cells[:i + 1]):
				self.gradients[-1].append(tf.gradients(cell.loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, cell2.name)))

	def load_weights(self, weights):
		for i, cell in enumerate(self.cells):
			cell.load_weights(weights)

	def fit(self, x, y):
		feed_dict = {}
		for i, cell in enumerate(self.cells):
			feed_dict[cell.input] = x[i]
			feed_dict[cell.labels] = y[i]
		feed_dict[self.cell_state] = np.zeros((5, self.units))
		feed_dict[self.hidden_state] = np.zeros((5, self.units))
		# loss, gradients = self.sess.run([self.loss, self.gradients], feed_dict)
		loss, gradients = self.sess.run([self.cells[-1].loss, self.gradients], feed_dict)
		return loss, gradients

class LSTMCell(object):

	def __init__(self, sess, units, cell_state, hidden_state, name):
		super(LSTMCell, self).__init__()
		self.units = units
		self.cell_state = cell_state
		self.hidden_state = hidden_state
		self.name = name
		self.sess = sess
		with tf.variable_scope(self.name):
			self.build_model()
			self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.weights_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in self.weights]
			assigns = [tf.assign(v, p) for v, p in zip(self.weights, self.weights_placeholders)]
			self.assign = tf.group(*assigns)

	def build_model(self):

		# input (per step)
		self.input = tf.placeholder(
			shape=[None, 1],
			dtype=tf.float32,
			name="{}_input".format(self.name),
		)

		# labels (per step)
		self.labels = tf.placeholder(
			shape=[None, self.units],
			dtype=tf.float32,
			name="{}_labels".format(self.name),
		)

		# forget gate
		self.forget_gate = tf.layers.dense(
			inputs=tf.concat([self.hidden_state, self.input], axis=1),
			units=self.units,
			activation=tf.sigmoid,
			kernel_initializer=tf.truncated_normal_initializer(.0,.01),
			name="forget_gate",
		)

		# input gate
		self.input_gate_filter = tf.layers.dense(
			inputs=tf.concat([self.hidden_state, self.input], axis=1),
			units=self.units,
			activation=tf.sigmoid,
			kernel_initializer=tf.truncated_normal_initializer(.0,.01),
			name="input_gate_filter",
		)
		self.input_gate_update = tf.layers.dense(
			inputs=tf.concat([self.hidden_state, self.input], axis=1),
			units=self.units,
			activation=tf.tanh,
			kernel_initializer=tf.truncated_normal_initializer(.0,.01),
			name="input_gate_update",
		)

		# output gate
		self.output_gate = tf.layers.dense(
			inputs=tf.concat([self.hidden_state, self.input], axis=1),
			units=self.units,
			activation=tf.sigmoid,
			kernel_initializer=tf.truncated_normal_initializer(.0,.01),
			name="output_gate",
		)

		# new cell state
		self.new_cell_state = self.forget_gate * self.cell_state + self.input_gate_filter * self.input_gate_update

		# new hidden state
		self.new_hidden_state = self.output_gate * tf.tanh(self.new_cell_state)

		self.output = self.new_hidden_state

		self.loss = tf.losses.mean_squared_error(self.labels, self.output)

	def load_weights(self, weights):
		feed_dict = dict(zip(self.weights_placeholders, weights))
		self.sess.run(self.assign, feed_dict=feed_dict)








