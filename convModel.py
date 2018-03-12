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

	min_word_freq = 4 ## Words with freq less than this are omitted from the vocabulary
	embed_size = 100
	hidden_layer_size = 128
	filter_sizes = [3,4,5,6]
	num_filters = 128
	label_size = 6
	max_epochs = 12
	batch_size = 64

	early_stopping = 5
	anneal_threshold = 3
	annealing_factor = 0.5
	lr = 1e-3
	l2 = 0.001

	model_name = 'model_RNN.weights'

class convModel(BaseModel):

	def define_weights(self):
		self.input_placeholder = tf.placeholder(tf.int32 , [None , None])
		self.label_placeholder = tf.placeholder(tf.float32 , [None , self.config.label_size])
		self.dropout_placeholder = tf.placeholder(tf.float32 )
		self.sequence_length_placeholder = tf.placeholder(tf.int32 , [None])

	def core_module(self , input_tensor):

		embed_size = self.config.embed_size
		hidden_layer_size = self.config.hidden_layer_size
		label_size = self.config.label_size
		filter_sizes = self.config.filter_sizes
		num_filters = self.config.num_filters

		total_pooled_op = []
		input_tensor = tf.expand_dims(input_tensor , axis=-1)
		max_len = tf.reduce_max(self.sequence_length_placeholder)

		for i,filter_size in enumerate(filter_sizes):

			with tf.variable_scope("conv_maxpool_%s"%i) as scope:
				W = tf.get_variable("Weight" , shape=[filter_size , embed_size, 1, num_filters] , initializer=tf.truncated_normal_initializer)
				bias = tf.get_variable("Bias" , shape=[num_filters])

				conv_op = tf.nn.conv2d(input_tensor , W, strides=[1,1,1,1], padding="VALID")
				filter_op = tf.nn.relu(tf.nn.bias_add(conv_op, bias))
				pool_op = tf.squeeze(tf.reduce_max(filter_op , axis=1))

				total_pooled_op.append(pool_op)

		pool_op = tf.nn.dropout(tf.concat(total_pooled_op , axis=1) , keep_prob=self.dropout_placeholder)

		with tf.variable_scope("FC_Layer") as scope:
			W_f = tf.get_variable("Weight" , shape=[num_filters*len(filter_sizes),hidden_layer_size] 
																	, initializer=tf.contrib.layers.xavier_initializer())
			b_f = tf.get_variable("Bias" , shape=[hidden_layer_size] , initializer=tf.zeros_initializer)

			W_o = tf.get_variable("Weight_o" , shape=[hidden_layer_size , label_size]
																	, initializer=tf.contrib.layers.xavier_initializer())
			b_o = tf.get_variable("Bias_o" , shape=[label_size] , initializer=tf.zeros_initializer)

		# fc_output = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(pool_op , W_f , b_f)) , keep_prob=self.dropout_placeholder)
		output = tf.nn.xw_plus_b(pool_op , W_o , b_o)

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

		y_input = y

		batch_word_len = X_input.shape[1]
		dropout = 1 if val else 0.9

		feed = {self.input_placeholder : X_input , self.dropout_placeholder : dropout}

		if y is not None:
			feed[self.label_placeholder] = y_input

		return feed