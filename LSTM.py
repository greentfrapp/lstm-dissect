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
			shape=[None, 10],
			dtype=tf.float32,
			name="main_labels"
		)
		self.loss = tf.losses.mean_squared_error(self.labels, self.cells[-1].output)
		self.lstm_gradients = []
		self.dense_gradients = []
		for i, cell in enumerate(self.cells):
			self.lstm_gradients.append([])
			self.dense_gradients.append(tf.gradients(cell.loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}/dense".format(cell.name))))
			for j, cell2 in enumerate(self.cells[:i + 1]):
				lstm_grads = tf.gradients(cell.loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}/lstm".format(cell2.name)))
				self.lstm_gradients[-1].append(lstm_grads)


	def load_weights(self, dense_weights, lstm_weights):
		for i, cell in enumerate(self.cells):
			cell.load_weights(dense_weights, lstm_weights)

	def fit(self, x, y):
		feed_dict = {}
		for i, cell in enumerate(self.cells):
			feed_dict[cell.input] = np.array(x)[:, i]
			feed_dict[cell.labels] = np.array(y)[:, i]
		feed_dict[self.cell_state] = np.zeros((len(x), self.units))
		feed_dict[self.hidden_state] = np.zeros((len(x), self.units))
		# loss, gradients = self.sess.run([self.loss, self.gradients], feed_dict)
		loss, dense_gradients, lstm_gradients = self.sess.run([self.cells[-1].loss, self.dense_gradients, self.lstm_gradients], feed_dict)
		return loss, dense_gradients, lstm_gradients

	def test(self, x):
		feed_dict = {}
		for i, cell in enumerate(self.cells):
			feed_dict[cell.input] = [x[i]]
		feed_dict[self.cell_state] = np.zeros((1, self.units))
		feed_dict[self.hidden_state] = np.zeros((1, self.units))
		outputs = []
		for cell in self.cells:
			outputs.append(cell.prediction)
		return self.sess.run(outputs, feed_dict)

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
			self.dense_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}/dense".format(self.name))
			self.dense_weights_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in self.dense_weights]
			self.lstm_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}/lstm".format(self.name))
			self.lstm_weights_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in self.lstm_weights]
			dense_assigns = [tf.assign(v, p) for v, p in zip(self.dense_weights, self.dense_weights_placeholders)]
			self.dense_assign = tf.group(*dense_assigns)
			lstm_assigns = [tf.assign(v, p) for v, p in zip(self.lstm_weights, self.lstm_weights_placeholders)]
			self.lstm_assign = tf.group(*lstm_assigns)

	def build_model(self):

		# input (per step)
		self.input = tf.placeholder(
			shape=[None, 1],
			dtype=tf.float32,
			name="{}_input".format(self.name),
		)

		# labels (per step)
		self.labels = tf.placeholder(
			shape=[None, 10],
			dtype=tf.float32,
			name="{}_labels".format(self.name),
		)

		with tf.variable_scope("lstm"):
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
				activation=tf.nn.relu,
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
			self.new_hidden_state = self.output_gate * tf.nn.relu(self.new_cell_state)

		# self.output = self.new_hidden_state
		# self.prediction = tf.sigmoid(self.output)

		with tf.variable_scope("dense"):
			output_dense_1 = tf.layers.dense(
				inputs=self.new_hidden_state,
				units=128,
				activation=tf.nn.relu,
				kernel_initializer=tf.truncated_normal_initializer(.0,.01),
				name="output_dense_1",
			)

			self.output = tf.layers.dense(
				inputs=output_dense_1,
				units=10,
				activation=None,
				kernel_initializer=tf.truncated_normal_initializer(.0,.01),
				name="output",
			)

		self.prediction = tf.nn.softmax(self.output)

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output))

	def load_weights(self, dense_weights, lstm_weights):
		feed_dict = dict(zip(self.dense_weights_placeholders, dense_weights))
		self.sess.run(self.dense_assign, feed_dict=feed_dict)
		feed_dict = dict(zip(self.lstm_weights_placeholders, lstm_weights))
		self.sess.run(self.lstm_assign, feed_dict=feed_dict)








