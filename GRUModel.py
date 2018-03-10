import tensorflow as tf
from LSTMModel import LSTMModel

class Config():

	min_word_freq = 2 ## Words with freq less than this are omitted from the vocabulary
	embed_size = 100
	hidden_size = 150
	label_size = 6
	max_epochs = 30
	batch_size = 128
	early_stopping = 5
	anneal_threshold = 3
	annealing_factor = 0.5
	lr = 1e-3
	l2 = 0.00

	model_name = 'model_RNN.weights'

class GRUModel(LSTMModel):

	def define_weights(self):
		embed_size = self.config.embed_size
		hidden_size = self.config.hidden_size
		label_size = self.config.label_size
		vocab_size = len(self.vocab)

		## Declare weights and placeholders
		with tf.variable_scope("Embeddings") as scope:
			embedding = tf.get_variable("Embeds", initializer=tf.random_uniform([vocab_size , embed_size] , -0.005,0.005) )

		with tf.variable_scope("Output" , initializer = tf.contrib.layers.xavier_initializer()) as scope:
			W_o = tf.get_variable("Weight" , [2*hidden_size , label_size])
			b_o = tf.get_variable("Bias" , [label_size])

		## Define the placeholders
		self.input_placeholder = tf.placeholder(tf.int32 , [None , None])
		self.label_placeholder = tf.placeholder(tf.float32 , [None , label_size])
		self.sequence_length_placeholder = tf.placeholder(tf.int32 , [None])

		self.cellstate_placeholder = tf.placeholder(tf.float32 , [None , hidden_size])
		self.hiddenstate_placeholder = tf.placeholder(tf.float32 , [None , hidden_size])

	def core_module(self , input_tensor):

		GRUcell = tf.contrib.rnn.GRUCell(num_units = self.config.hidden_size)
		last_hiddenstate_fwd = tf.nn.dynamic_rnn(GRUcell , input_tensor , sequence_length=self.sequence_length_placeholder,	
														  		initial_state=self.hiddenstate_placeholder , scope="fwd")[1]

		GRUcell_rev = tf.contrib.rnn.GRUCell(num_units = self.config.hidden_size)
		input_rev = tf.reverse(input_tensor , axis=[1])
		last_hiddenstate_rev = tf.nn.dynamic_rnn(GRUcell_rev , input_rev ,	
														  		initial_state=self.hiddenstate_placeholder , scope="rev")[1]
		last_hiddenstate = tf.concat([last_hiddenstate_fwd , last_hiddenstate_rev] , axis=1)


		with tf.variable_scope("Output" , reuse=True):
			W_o = tf.get_variable("Weight")
			b_o = tf.get_variable("Bias")

			output = tf.matmul(last_hiddenstate , W_o) + b_o

		return output

	def __init__(self , config , datafile , debug=False):
		super().__init__(config , datafile , debug)

if __name__ == "__main__":
	print('Import libraries. Not to be run separately.')