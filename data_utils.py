from collections import defaultdict
import re
import numpy as np
import pandas as pd

unknown_string = 'UNNKKK'

def get_words(line):
	line = line.lower()
	return re.split('\W+|_' , line) ## Return ONLY words. Split the sentence by anything which isn't a word

class Vocab():

	def __init__(self):

		self.word_to_idx = {}
		self.idx_to_word = {}
		self.word_freq = defaultdict(int)
		self.total_words = 0 #float(sum(self.word_data.values()))
		self.unknown = unknown_string
		self.add_word(self.unknown , count=0)

	def add_word(self ,word , count=1):
		if word not in word_to_idx:
			index = len(self.word_to_idx)
			self.word_to_idx[word] = index
			self.idx_to_word[index] = word
		self.word_freq[word] += count

	def construct(self, words_list , replace_digits=True):
		for word,count in words_list.items():
			if any([c.isdigit() for c in word]) and replace_digits:
				word = self.unknown
			self.add_word(word , count)
		total_words = float(sum(self.word_freq.values()))
		print('Total {} words with {} uniques'.format(total_words , len(self.word_freq)) )

	def encode(self, word):
		if word not in word_to_idx:
			word = self.unknown
		return word_to_idx[word]

	def decode(self , index):
		return idx_to_word[index]

	def __len__(self):
		return len(self.word_to_idx)


def get_batches(X, y=None , batch_size=1 , augment_method='pad' , common_size=10):

	num_batches = int(len(X)/batch_size)
	augment_method = augment_method.lower()

	## Sort the entries in data with length of sentences to reduce computation time.
	sort_idx = [val[0] for val in sorted(enumerate(X) , key = lambda K : len(get_words(K[1])))]
	X = X[sort_idx]
	y = y[sort_idx]

	if y is None:
		get_labels = False

	X_data = []
	y_data = []
	seq_lengths = []

	for idx in num_batches:
		sents = np.reshape(X.loc[idx*batch_size:(idx+1)*batch_size] , [-1,1])

		if get_labels:
			labels = y.loc[idx*batch_size:(idx+1)*batch_size , :]
			y_data.append(labels)

		batch_words = [get_words(sent) for sent in sents]
		seq_lengths.append([len(batch) for batch in batch_words])
		modified_words = []

		if augment_method == 'pad': ## Match the maximum
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
				elif:
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

	if get_labels:
		return X_data, seq_lengths , y_data
	return X_data , seq_lengths

