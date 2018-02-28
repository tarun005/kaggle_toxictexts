class Config():

	embed_size = 60
	hidden_size = 100
	label_size = 5
	num_epochs = 30
	batch_size = 32
	early_stopping = 5
	anneal_threshold = 3
	anneal_factor = 0.5
	lr = 0.001
	l2 = 0.01

	model_name = 'model_RNN.weights'

class RNNModel():

	def load_data(self , datafile):

		dataset = pd.read_csv('train.csv')
		text = 'comment_text'
		self.X = dataset[text].values

		if self.test_mode:
			self.X_test = self.X
			return
		
		labels = ['toxic', 'severe_toxic', 'obscene' , 'threat', 'insult', 'identity_hate']
		self.y = dataset[labels].values
		self.X_train , self.X_val , self.y_train , self.y_val = train_test_split(self.X , self.y, test_size=0.1, random_state=1234)

		## Build the vocabulary using the train data.
		self.vocab = Vocab()
		train_sents = [get_words(line) for line in self.X_train]
		all_words = list(itertools.chain.from_iterable(train_sents))
		unique_words , n_occs = np.unique(all_words , return_counts=True) ## Get unique tokens
		words_freq_dictionary = dict(zip(unique_words , n_occs))
		self.vocab.construct(words_freq_dictionary)

	def define_weights(self):
		embed_size = self.config.embed_size
		hidden_size = self.config.hidden_size
		label_size = self.config.label_size
		vocab_size = len(self.vocab)

		## Declare weights and placeholders
		with tf.variable_scope("Embeddings" , initializer = tf.contrib.layers.xavier_initializer) as scope:
			embedding = tf.get_variable("Embeds" , [vocab_size , embed_size])

		with tf.variable_scope("Output" , initializer = tf.contrib.layers.xavier_initializer) as scope:
			W_o = tf.get_variable("Weight" , [hidden_size , label_size])
			b_o = tf.get_variable("Bias" , [label_size])
			self.wo_l2loss = tf.nn.l2_loss(W_o)

		## Define the placeholders
		self.input_placeholder = tf.placeholder(tf.int32 , [None , None])
		self.label_placeholder = tf.placeholder(tf.int32 , [None , label_size])
		self.sequence_length_placeholder = tf.placeholder(tf.int32 , [None])

		self.cellstate_placeholder = tf.placeholder(tf.float32 , [None , hidden_size])
		self.hiddenstate_placeholder = tf.placeholder(tf.float32 , [None , hidden_size])

	def input_embeddings(self):

		with tf.variable_scope("Embeddings"):
			embedding = tf.get_variable("Embeds")

		input_vectors = tf.nn.embedding_lookup(embedding , self.input_placeholder)
		num_splits = input_vectors.get_shape().as_list()[1]
		input_series = [tf.squeeze(input_step_mat,axis=1) for input_step_mat in tf.split(input_vectors , num_splits, axis=1)]

		return input_series

	def LSTM_module(self , input_series):

		state_tuple = tf.contrib.rnn.LSTMStateTuple(self.cellstate_placeholder , self.hiddenstate_placeholder)
		LSTMcell = tf.contrib.rnn.BasicLSTMCell (num_units = self.config.hidden_size , state_is_tuple=True)
		hidden_states , last_state = tf.nn.static_rnn(LSTMcell , input_series ,
														   state_tuple,	sequence_length=self.sequence_length_placeholder)
		last_cellstate , last_hiddenstate = last_state

		with tf.variable_scope("Output"):
			W_o = tf.get_variable("Weight")
			b_o = tf.get_variable("Bias")

			output = tf.matmul(W_o , last_hiddenstate) + b_o

		return output

	def calculate_loss(self , output):

		labels = self.label_placeholder

		log_loss = tf.reduce_mean(tf.multiply(labels , tf.log_sigmoid(output)) +
												 tf.multiply((1-labels) , tf.log_sigmoid(-1*output)))
		l2_loss = 0
		for weights in tf.trainable_variables():
			if ("Bias" not in weights.name) and ("Embeddings" not in weights.name): 
				l2_loss += self.config.l2_loss(weights)

		loss = log_loss #+ l2_loss

		return loss

	def training_operation(self , loss):
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)

	def __init__(self , config , datafile , test=False):
		self.config = config
		self.test_mode = test
		self.load_data(datafile)
		self.define_weights()
		input_series = self.input_embeddings()
		output = self.LSTM_module(input_series)
		self.loss = self.calculate_loss(output)
		self.training_operation(loss)

		self.pred = tf.cast(tf.greater(tf.nn.sigmoid(output) , 0.5) , tf.int32)

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

		feed = {self.input_placeholder : X_input,
				self.sequence_length_placeholder : seq_len_input,
				self.label_placeholder : y_input,
				self.cellstate_placeholder : np.zeros([self.config.batch_size , self.config.hidden_size]),
				self.hiddenstate_placeholder : np.zeros([self.config.batch_size , self.config.hidden_size]),
				}

		return feed