import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from data_utils import Vocab
from data_utils import get_words

class BaseModel():

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

		## Declare weights and placeholders
		with tf.variable_scope("Embeddings" , initializer = tf.contrib.layers.xavier_initializer()) as scope:
			embedding = tf.get_variable("Embeds" , shape=[1,1])

		## Define the placeholders
		self.input_placeholder = tf.placeholder(tf.int32 , [None , None])
		self.label_placeholder = tf.placeholder(tf.float32 , [None , label_size])

	def input_embeddings(self):

		with tf.variable_scope("Embeddings" , reuse=True):
			embedding = tf.get_variable("Embeds")

		input_vectors = tf.nn.embedding_lookup(embedding , self.input_placeholder)
		return input_vectors

	def core_module(self):

		return

	def calculate_loss(self , output):

		labels = self.label_placeholder

		log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output , labels=labels))

		l2_loss = 0
		for weights in tf.trainable_variables():
			if ("Bias" not in weights.name) and ("Embeddings" not in weights.name): 
				l2_loss += (self.config.l2 * tf.nn.l2_loss(weights))

		loss = log_loss + l2_loss

		return loss

	def training_operation(self , loss):
		return tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)

	def build_feeddict(self):

		return

if __name__ == "__main__":
	print('Import libraries. Not to be run separately.')