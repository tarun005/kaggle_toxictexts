from collections import defaultdict
import re
import numpy as np
from sklearn.metrics import roc_auc_score

unknown_string = 'UNNKKK'

def get_words(line):
	line = line.lower()
	return re.split('\s+|_' , line) ## Return ONLY words.

class Vocab():

	def __init__(self):

		self.word_to_idx = {}
		self.idx_to_word = {}
		self.word_freq = defaultdict(int)
		self.total_words = 0 
		self.unknown = unknown_string
		self.add_word(self.unknown , count=0)

	def add_word(self ,word , count=1):
		if word not in self.word_to_idx:
			index = len(self.word_to_idx)
			self.word_to_idx[word] = index
			self.idx_to_word[index] = word
		self.word_freq[word] += count

	def construct(self, words_list ,threshold=5, replace_digits=True):
		for word in words_list:
			if any([c.isdigit() for c in word]) and replace_digits:
				word = self.unknown
			self.add_word(word)
		self.total_words = sum(self.word_freq.values())
		self.trim_vocab(threshold)
		print('Total {} words with {} uniques'.format(self.total_words , len(self.word_freq)) )

		assert (len(self.word_to_idx) == len(self.idx_to_word))
		assert(len(self.word_to_idx) == len(self.word_freq))

	def trim_vocab(self , threshold):
		self.idx_to_word = {}
		temp_dict = dict(self.word_freq) ## Create a copy to avoid binding
		for word , count in self.word_freq.items():
			if count < threshold and word != unknown_string:
				temp_dict[self.unknown] += temp_dict[word]
				del self.word_to_idx[word]
				del temp_dict[word]
		self.word_freq = dict(temp_dict)

		## Reindex the words.
		self.word_to_idx = dict(zip(self.word_to_idx.keys() , range(len(self.word_to_idx))))
		self.idx_to_word = {v:k for k,v in self.word_to_idx.items()}

	def encode(self, word):
		if word not in self.word_to_idx:
			word = self.unknown
		return self.word_to_idx[word]

	def decode(self , index):
		return idx_to_word[index]

	def __len__(self):
		return len(self.word_to_idx)


def get_batches(X, y=None , batch_size=1 ,shuffle=True, augment_method='pad' , common_size=100):

	num_batches = int(len(X)/batch_size)
	augment_method = augment_method.lower()

	get_labels = False if y is None else True

	# Sort the entries in data with length of sentences to reduce computation time.
	sort_idx = [val[0] for val in sorted(enumerate(X) , key = lambda K : len(get_words(K[1])))]
	X = X[sort_idx]
	y = y[sort_idx] if get_labels else None

	X_data = []
	y_data = []
	seq_lengths = []

	for idx in range(num_batches):
		sents = np.reshape(X[idx*batch_size:(idx+1)*batch_size] , [-1,1])

		if get_labels:
			labels = y[idx*batch_size:(idx+1)*batch_size , :]
			y_data.append(labels)

		batch_words = [get_words(sent[0]) for sent in sents] ## Array to list
		seq_lengths.append([len(batch) for batch in batch_words])
		modified_words = []

		if augment_method == 'pad': ## Match the maximum of batch; default.
			max_len = max([len(batch) for batch in batch_words])
			modified_words = []
			for batch in batch_words:
				len_batch = len(batch)
				all_words = batch + [unknown_string]*(max_len - len_batch)
				modified_words.append(all_words)

		elif augment_method == 'adjust': ## Adjust to a fixed length
			fixed_len = common_size
			modified_words = []
			for batch in batch_words:
				len_batch = len(batch)
				if len_batch < common_size:
					all_words = batch + [unknown_string]*(fixed_len - len_batch)
				else:
					all_words = batch[:common_size]
				modified_words.append(all_words)

		elif augment_method == 'truncate': ## Match the minimum
			min_len = min([len(batch) for batch in batch_words])
			modified_words = []
			for batch in batch_words:
				all_words = batch[:common_size]
				modified_words.append(all_words)

		else:
			raise ValueError("Augment Method not specified or not understood. Should be one of 'pad' , 'adjust' or 'truncate'")

		X_data.append(modified_words)

	r_idx = np.random.permutation(np.arange(len(X_data))) if shuffle else np.arange(len(X_data))

	assert(len(X_data) == len(seq_lengths))

	X_data = [X_data[idx] for idx in r_idx]
	seq_lengths = [seq_lengths[idx] for idx in r_idx]

	if not get_labels:
		return X_data , seq_lengths

	y_data = [y_data[idx] for idx in r_idx]
	return X_data, seq_lengths, y_data

def accuracy(labels , predictions , classwise=True):

	roc_auc = []
	for col in range(labels.shape[1]):
		roc_auc.append(roc_auc_score(labels[:,col] , predictions[:,col]))

	roc_auc = np.round(roc_auc , 4)
	if classwise:
		return roc_auc
	else:
		return np.mean(roc_auc)

if __name__ == "__main__":
	print('Import libraries. Not to be run separately.')

