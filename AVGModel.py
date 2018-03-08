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

	min_word_freq = 2 ## Words with freq less than this are omitted from the vocabulary
	embed_size = 100
	hidden_size_1 = 512
	hidden_size_2 = 256
	label_size = 6
	max_epochs = 30
	batch_size = 64
	early_stopping = 5
	anneal_threshold = 3
	annealing_factor = 0.5
	lr = 3e-4
	l2 = 0.001

	model_name = 'model_RNN.weights'

class AVGModel(BaseModel):

	def define_weights(self):
		embed_size = self.config.embed_size
		hidden_size_1 = self.config.hidden_size_1
		hidden_size_2 = self.config.hidden_size_2
		label_size = self.config.label_size
		vocab_size = len(self.vocab)

		## Declare weights and placeholders
		with tf.variable_scope("Embeddings" , initializer = tf.contrib.layers.xavier_initializer()) as scope:
			embedding = tf.get_variable("Embeds" , shape=[vocab_size , embed_size])

		with tf.variable_scope("Neural" , initializer = tf.contrib.layers.xavier_initializer()) as scope:
			W_1 = tf.get_variable("Weight_1" , [embed_size , hidden_size_1])
			b_1 = tf.get_variable("Bias_1" , [hidden_size_1])

			W_2 = tf.get_variable("Weight_2" , [hidden_size_1 , hidden_size_2])
			b_2 = tf.get_variable("Bias_2" , [hidden_size_2])

		with tf.variable_scope("Output" , initializer = tf.contrib.layers.xavier_initializer()) as scope:
			W_o = tf.get_variable("Weight" , [hidden_size_2 , label_size])
			b_o = tf.get_variable("Bias" , [label_size])
			self.wo_l2loss = tf.nn.l2_loss(W_o)

		## Define the placeholders
		self.input_placeholder = tf.placeholder(tf.int32 , [None , None])
		self.label_placeholder = tf.placeholder(tf.float32 , [None , label_size])

	def core_module(self , input_tensor):

		input_tensor = tf.reduce_mean(input_tensor , axis=1, keep_dims=False)

		with tf.variable_scope("Neural" ,reuse=True):
			W_1 = tf.get_variable("Weight_1")
			b_1 = tf.get_variable("Bias_1" )

			W_2 = tf.get_variable("Weight_2")
			b_2 = tf.get_variable("Bias_2")

		hidden_layer_1 = tf.nn.relu(tf.matmul(input_tensor , W_1) + b_1)
		hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1 , W_2) + b_2)
		
		with tf.variable_scope("Output" , reuse=True):
			W_o = tf.get_variable("Weight")
			b_o = tf.get_variable("Bias")

			output = tf.matmul(hidden_layer_2 , W_o) + b_o

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

	def build_feeddict(self, X, seq_len=None, y=None):

		X_input = []

		for sent in X:
			sent_tokens = []
			for word in sent:
				sent_tokens.append(self.vocab.encode(word))
			X_input.append(sent_tokens)

		X_input = np.array(X_input)
		y_input = y

		batch_word_len = X_input.shape[1]

		feed = {self.input_placeholder : X_input}

		if y is not None:
			feed[self.label_placeholder] = y_input

		return feed

if __name__ == "__main__":
	print('Import libraries. Not to be run separately.')