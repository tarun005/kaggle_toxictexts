import tensorflow as tf
from LSTMModel import LSTMModel

class Config():

	min_word_freq = 5 ## Words with freq less than this are omitted from the vocabulary
	embed_size = 300
	hidden_size = 256
	hidden_layer_size = 128
	label_size = 6
	max_epochs = 6
	batch_size = 64
	early_stopping = 5
	anneal_threshold = 3
	annealing_factor = 0.5
	lr = 1e-3
	l2 = 0.00
	dropout = 0.8

	model_name = 'model_RNN.weights'

class GRUModel(LSTMModel):

	def define_weights(self):
		embed_size = self.config.embed_size
		hidden_size = self.config.hidden_size
		label_size = self.config.label_size

		## Declare weights and placeholders

		with tf.variable_scope("Output" , initializer = tf.contrib.layers.xavier_initializer()) as scope:
			W_1 = tf.get_variable("Weight_1" , [4*hidden_size , self.config.hidden_layer_size])
			b_1 = tf.get_variable("Bias_1" , [self.config.hidden_layer_size])

			W_o = tf.get_variable("Weight" , [self.config.hidden_layer_size , label_size])
			b_o = tf.get_variable("Bias" , [label_size])

		## Define the placeholders
		self.input_placeholder = tf.placeholder(tf.int32 , [None , None])
		self.label_placeholder = tf.placeholder(tf.float32 , [None , label_size])
		self.sequence_length_placeholder = tf.placeholder(tf.int32 , [None])

		self.hiddenstate_placeholder = tf.placeholder(tf.float32 , [None , hidden_size])
		self.cellstate_placeholder = tf.placeholder(tf.float32 , [None , hidden_size])

		self.dropout_placeholder = tf.placeholder(tf.float32 )

	def core_module(self , input_tensor):

		seq_len = self.sequence_length_placeholder
		init_state = self.hiddenstate_placeholder

		with tf.variable_scope("First_LSTM") as scope:
			LSTMcell_fwd_1 = tf.contrib.rnn.GRUCell(num_units = self.config.hidden_size)
			fwd_seq_list_1, _ = tf.nn.dynamic_rnn(LSTMcell_fwd_1 , input_tensor ,sequence_length=seq_len, initial_state=init_state , scope="Forward")

			LSTMcell_rev_1 = tf.contrib.rnn.GRUCell(num_units = self.config.hidden_size)
			reverse_input = tf.reverse(input_tensor , axis=[1])
			rev_seq_list_1 , _ = tf.nn.dynamic_rnn(LSTMcell_rev_1 , reverse_input, initial_state=init_state , scope="Backward")

		# with tf.variable_scope("Second_LSTM") as scope:
		# 	LSTMcell_fwd_2 = tf.contrib.rnn.GRUCell(num_units = self.config.hidden_size)
		# 	fwd_seq_list_2, _ = tf.nn.dynamic_rnn(LSTMcell_fwd_2 , fwd_seq_list_1 ,sequence_length=seq_len, initial_state=init_state , scope="Forward")
			
		# 	LSTMcell_rev_2 = tf.contrib.rnn.GRUCell(num_units = self.config.hidden_size)
		# 	rev_seq_list_2 , _ = tf.nn.dynamic_rnn(LSTMcell_rev_2 , rev_seq_list_1, initial_state=init_state , scope="Backward")
			

		hiddenstate_fwd = tf.reduce_max(fwd_seq_list_1 , axis=1 , keep_dims=False)
		hiddenstate_fwd_mean = tf.reduce_mean(fwd_seq_list_1 , axis=1 , keep_dims=False)

		hiddenstate_rev = tf.reduce_max(rev_seq_list_1 , axis=1 , keep_dims=False)
		hiddenstate_rev_mean = tf.reduce_mean(rev_seq_list_1 , axis=1 , keep_dims=False)

		last_hiddenstate = tf.concat([hiddenstate_fwd, hiddenstate_fwd_mean , hiddenstate_rev, hiddenstate_rev_mean],axis=1)

		with tf.variable_scope("Output" , reuse=True):
			W_1 = tf.get_variable("Weight_1")
			b_1 = tf.get_variable("Bias_1")
			W_o = tf.get_variable("Weight")
			b_o = tf.get_variable("Bias")

		hidden_output = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(last_hiddenstate , W_1 , b_1)) , keep_prob=self.dropout_placeholder)
		output = tf.nn.xw_plus_b(hidden_output , W_o , b_o)

		return output

	def __init__(self , config , datafile , debug=False):
		super().__init__(config , datafile , debug)

if __name__ == "__main__":
	print('Import libraries. Not to be run separately.')