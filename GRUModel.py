import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import os , sys , time
from data_utils import Vocab
from data_utils import get_words

class Config():

	min_word_freq = 2 ## Words with freq less than this are omitted from the vocabulary
	embed_size = 100
	hidden_size = 150
	label_size = 6
	max_epochs = 30
	batch_size = 64
	early_stopping = 5
	anneal_threshold = 3
	annealing_factor = 0.5
	lr = 1e-3
	l2 = 0.001

	model_name = 'model_RNN.weights'

class RNNModel():

	def load_data(self , datafile):

		dataset = pd.read_csv(datafile)
		if self.debug:
			dataset = dataset.iloc[:1000]
			
		text = 'comment_text'
		self.X = dataset[text].values
		
		labels = ['toxic', 'severe_toxic', 'obscene' , 'threat', 'insult', 'identity_hate']
		# labels = ['severe_toxic']
		assert(len(labels) == self.config.label_size)
		self.y = dataset[labels].values
		self.X_train , self.X_val , self.y_train , self.y_val = train_test_split(self.X , self.y, test_size=0.1, random_state=1234)

		## Build the vocabulary using the train data.
		self.vocab = Vocab()
		train_sents = [get_words(line) for line in self.X_train]
		self.vocab.construct(list(itertools.chain.from_iterable(train_sents)) , threshold=self.config.min_word_freq)
		print('Training on {} samples and validating on {} samples'.format(len(self.X_train) , len(self.X_val)))
		print()

	def define_weights(self):
		embed_size = self.config.embed_size
		hidden_size = self.config.hidden_size
		label_size = self.config.label_size
		vocab_size = len(self.vocab)

		## Declare weights and placeholders
		with tf.variable_scope("Embeddings" , initializer = tf.contrib.layers.xavier_initializer()) as scope:
			embedding = tf.get_variable("Embeds" , shape=[vocab_size , embed_size])

		with tf.variable_scope("Output" , initializer = tf.contrib.layers.xavier_initializer()) as scope:
			W_o = tf.get_variable("Weight" , [hidden_size , label_size])
			b_o = tf.get_variable("Bias" , [label_size])
			self.wo_l2loss = tf.nn.l2_loss(W_o)

		## Define the placeholders
		self.input_placeholder = tf.placeholder(tf.int32 , [None , None])
		self.label_placeholder = tf.placeholder(tf.float32 , [None , label_size])
		self.sequence_length_placeholder = tf.placeholder(tf.int32 , [None])
		self.hiddenstate_placeholder = tf.placeholder(tf.float32 , [None , hidden_size])

	def input_embeddings(self):

		with tf.variable_scope("Embeddings" , reuse=True):
			embedding = tf.get_variable("Embeds")

		input_vectors = tf.nn.embedding_lookup(embedding , self.input_placeholder)
		return input_vectors

	def core_module(self , input_tensor):

		GRUcell = tf.contrib.rnn.tf.contrib.rnn.GRUCell(num_units = self.config.hidden_size)
		last_hiddenstate = tf.nn.dynamic_rnn(GRUcell , input_tensor , sequence_length=self.sequence_length_placeholder,	
														  		initial_state=self.hiddenstate_placeholder)[1]

		with tf.variable_scope("Output" , reuse=True):
			W_o = tf.get_variable("Weight")
			b_o = tf.get_variable("Bias")

			output = tf.matmul(last_hiddenstate , W_o) + b_o

		return output

	def calculate_loss(self , output):

		labels = self.label_placeholder

		log_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output , labels=labels)

		l2_loss = 0
		for weights in tf.trainable_variables():
			if ("Bias" not in weights.name) and ("Embeddings" not in weights.name): 
				l2_loss += (self.config.l2 * tf.nn.l2_loss(weights))

		loss = log_loss + l2_loss

		return loss

	def training_operation(self , loss):
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)

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
		self.training_operation(self.loss)

		self.pred = tf.cast(tf.greater(tf.nn.sigmoid(output) , 0.5) , tf.float32)

	def build_feeddict(self, X, seq_len, y=None):

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