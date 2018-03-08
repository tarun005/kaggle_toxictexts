import tensorflow as tf
from LSTMModel import LSTMModel

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

class GRUModel(LSTMModel):

	def core_module(self , input_tensor):

		GRUcell = tf.contrib.rnn.GRUCell(num_units = self.config.hidden_size)
		last_hiddenstate = tf.nn.dynamic_rnn(GRUcell , input_tensor , sequence_length=self.sequence_length_placeholder,	
														  		initial_state=self.hiddenstate_placeholder)[1]


		with tf.variable_scope("Output" , reuse=True):
			W_o = tf.get_variable("Weight")
			b_o = tf.get_variable("Bias")

			output = tf.matmul(last_hiddenstate , W_o) + b_o

		return output

	def __init__(self , config , datafile , debug=False):
		super().__init__(config , datafile , debug)

if __name__ == "__main__":
	print('Import libraries. Not to be run separately.')