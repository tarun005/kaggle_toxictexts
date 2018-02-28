import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import os , sys , time
from model import RNNModel , Config
from data_utils import get_batches, get_words
from data_utils import Vocab

def accuracy(labels , predictions , classwise=True):

	if classwise:
		class_wise_acc = np.mean(np.equal(labels , predictions) , axis=0, keepdims=True)
		return class_wise_acc
	else:
		acc = np.mean(np.equal(labels , predictions))
		return acc

def run_epoch(sess , model, verbose=True):

	epoch_train_pred = []
	epoch_train_label = []
	epoch_train_loss = []

	epoch_val_pred = []
	epoch_val_label = []
	epoch_val_loss = []

	step = 0

	for train_X, train_seq_len, train_y in get_batches(model.X_train , model.y_train, model.config.batch_size):
		feed = model.build_feeddict(train_X , train_seq_len, train_y) 
		class_pred , batch_loss , _ = sess.run([model.pred , model.loss, model.train_op] , feed_dict=feed)
		epoch_train_pred.append(class_pred)
		epoch_train_label.append(train_y)
		epoch_train_loss.append(batch_loss)

		if verbose:
			sys.stdout.write('\r{} / {} :    loss = {}'.format(step*self.config.batch_size, len(model.X_train), np.mean(epoch_train_loss)))
			sys.stdout.flush()

		step += 1

	for val_X , val_seq_len, val_y in get_batches(model.X_val, model.y_val, model.config.batch_size):
		feed = model.build_feeddict(val_X , val_seq_len, val_y)
		val_preds , val_loss = sess.run([model.pred , model.loss] , feed_dict=feed)
		epoch_val_pred.append(val_preds)
		epoch_val_label.append(val_y)
		epoch_val_loss.append(val_loss)


	train_predictions = np.concatenate(epoch_train_pred , axis=0)
	train_labels = np.concatenate(epoch_train_label , axis=0)
	train_acc = accuracy(labels , predictions) 

	val_predictions = np.concatenate(epoch_val_pred , axis=0)
	val_labels = np.concatenate(epoch_val_label , axis=0)
	val_acc = accuracy(labels , predictions) 

	print("Training Accuracy: {}".format(train_acc))
	print("Validation Accuracy: {}".format(val_acc))
	print()

	return epoch_train_loss, epoch_val_loss, train_acc , val_acc

def train_RNNModel(filename):

	start_time = time.time()

	config = Config()
	model = RNNModel(config , filename)

	num_batches = int(len(model.X_train)/batch_size)
	train_loss_history = np.zeros(model.config.num_epochs , model.config.num_batches)
	val_loss_history = np.zeros_like(train_loss_history)
	train_acc_history = np.zeros(model.config.num_epochs , model.config.label_size)
	val_acc_history = np.zeros_lke(train_acc_history)

	best_val_loss = np.float(inf)
	best_epoch = 0

	if not os.path.exists("./weights"):
		os.makedirs("./weights")

	with tf.Session() as sess:

		init = tf.global_variables_initalizer()
		sess.run(init)

		for epoch in num_epochs:

			epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc = run_epoch(sess , model)
			train_acc_history[epoch , :] = epoch_train_acc
			val_acc_history[epoch , :] = epoch_val_acc
			train_loss_history[epoch , :] = np.array(epoch_train_loss)
			val_loss_history[epoch , :] = np.array(epoch_val_loss)

			val_loss = np.mean(epoch_val_loss)

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				best_epoch = epoch
				saver = tf.train.Saver()
				saver.save(sess , './weights/%s'%model.config.model_name)

			if epoch - best_epoch > model.config.anneal_threshold: ## Anneal lr on no improvement in val loss
				model.config.lr *= model.config.annealing_factor

			if epoch - best_epoch > model.config.early_stopping: ## Stop on no improvement
				print('Stopping due to early stopping')
				break;


	print()
	print("#"*20)
	print('Completed Training')
	print('Training Time:{} minutes'.format((time.time()-start_time)/60))

	plt.plot(np.mean(train_loss_history , axis=0) , linewidth=3 , label='Train')
	plt.plot(np.mean(val_loss_history , axis=0) , linewidth=3 , label='Val')
	plt.xlabel("Epoch Number")
	plt.ylabel("Loss")
	plt.title("Loss vs Epoch")
	plt.legend()

def test_RNNModel(filename):

	test_data = pd.read_csv(filename)
	test_idx = test_data.iloc[:,0].values

	config = Config()
	model = RNNModel(config , filename , test=True) ## Builds our model

	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('./weights/%s.meta'%model.config.model_name)
		saver.restore(sess , './weights/%s'%model.config.model_name)

		X_test , test_seq_length = get_batches(model.X_test, y=None, batch_size=1)
		feed = model.build_feeddict(X_test , test_seq_length)
		predictions = sess.run([model.pred] , feed_dict=feed)

	assert(len(test_idx) == len(predictions))

	## Code to write the output submissions to a file

if __name__ == "__main__":

	train_RNNModel(filename = 'train.txt') ## Save the weights and model
	# test_RNNModel(filename = 'test.txt') ## Load the model and test