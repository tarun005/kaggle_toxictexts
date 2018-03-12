import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import os , sys , time
from data_utils import Vocab
from data_utils import get_words
from BaseModel import BaseModel

class Config():

	min_word_freq = 0 ## Words with freq less than this are omitted from the vocabulary
	embed_size = 100
	hidden_size = 100
	hidden_size_output = 64
	label_size = 6
	max_epochs = 8
	batch_size = 64
	early_stopping = 5
	anneal_threshold = 3
	annealing_factor = 0.5
	lr = 1e-3
	l2 = 0.00

	model_name = 'model_RNN.weights'

class LSTMModel(BaseModel):

	def define_weights(self):
		embed_size = self.config.embed_size
		hidden_size = self.config.hidden_size
		label_size = self.config.label_size
		vocab_size = len(self.vocab)

		## Declare weights and placeholders

		with tf.variable_scope("Output" , initializer = tf.contrib.layers.xavier_initializer()) as scope:
			W_o = tf.get_variable("Weight" , [2*hidden_size , label_size])
			b_o = tf.get_variable("Bias" , [label_size] , initializer=tf.zeros_initializer)

		## Define the placeholders
		self.input_placeholder = tf.placeholder(tf.int32 , [None , None])
		self.label_placeholder = tf.placeholder(tf.float32 , [None , label_size])
		self.sequence_length_placeholder = tf.placeholder(tf.int32 , [None])

		self.cellstate_placeholder = tf.placeholder(tf.float32 , [None , hidden_size])
		self.hiddenstate_placeholder = tf.placeholder(tf.float32 , [None , hidden_size])

	def core_module(self , input_tensor):

		seq_len = self.sequence_length_placeholder
		state_tuple = tf.contrib.rnn.LSTMStateTuple(self.cellstate_placeholder , self.hiddenstate_placeholder)

		LSTMcell_fwd = tf.contrib.rnn.BasicLSTMCell(num_units = self.config.hidden_size)
		fwd_seq_list, _ = tf.nn.dynamic_rnn(LSTMcell_fwd , input_tensor ,sequence_length=seq_len, initial_state=state_tuple , scope="Forward")
		last_hiddenstate_fwd = tf.reduce_max(fwd_seq_list , axis=1 , keep_dims=False)

		LSTMcell_rev = tf.contrib.rnn.BasicLSTMCell(num_units = self.config.hidden_size)
		reverse_input = tf.reverse(input_tensor , axis=[1])
		rev_seq_list , _ = tf.nn.dynamic_rnn(LSTMcell_rev , reverse_input, initial_state=state_tuple , scope="Backward")
		last_hiddenstate_rev = tf.reduce_max(rev_seq_list , axis=1 , keep_dims=False)

		last_hiddenstate = tf.concat([last_hiddenstate_fwd , last_hiddenstate_rev] , axis=1)

		with tf.variable_scope("Output" , reuse=True):
			W_o = tf.get_variable("Weight")
			b_o = tf.get_variable("Bias")

		output = tf.matmul(last_hiddenstate , W_o) + b_o

		return output

	def __init__(self , config , datafile , debug=False):
		self.config = config
		if debug:
			self.config.max_epochs = 1
		self.debug = debug
		self.load_data(datafile)
		self.define_weights() 
		input_tensor = self.input_embeddings()
		output = self.core_module(input_tensor)
		self.loss = self.calculate_loss(output)
		self.train_op = self.training_operation(self.loss)

		self.pred = tf.nn.sigmoid(output)

	def build_feeddict(self, X, seq_len, y=None, val=False):

		X_input = []

		for sent in X:
			sent_tokens = []
			for word in sent:
				sent_tokens.append(self.vocab.encode(word))
			X_input.append(sent_tokens)

		X_input = np.array(X_input)
		assert(X_input.shape[0] == self.config.batch_size)

		y_input = y
		seq_len_input = np.reshape(np.array(seq_len) , [-1])
		assert(len(seq_len_input) == self.config.batch_size)

		batch_word_len = X_input.shape[1]

		feed = {self.input_placeholder : X_input,
				self.sequence_length_placeholder : seq_len_input,
				self.cellstate_placeholder : np.zeros([self.config.batch_size , self.config.hidden_size]),
				self.hiddenstate_placeholder : np.zeros([self.config.batch_size , self.config.hidden_size]),
				}

		if y is not None:
			feed[self.label_placeholder] = y_input

		return feed

if __name__ == "__main__":
	print('Import libraries. Not to be run separately.')