import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
import pandas as pd
import sys, time
from LSTMModel import RNNModel as Model , Config
from data_utils import get_batches, get_words, accuracy
from data_utils import Vocab
import importlib
import matplotlib.pyplot as plt

def run_epoch(sess , model, iter_obj, val=False, verbose=True):

	epoch_pred = []
	epoch_label = []
	epoch_loss = []

	step = 0

	for X, seq_len, y in iter_obj:
		feed = model.build_feeddict(X , seq_len, y)
		if val:
			class_pred , batch_loss = sess.run([model.pred , model.loss] , feed_dict=feed)
		else:
			class_pred , batch_loss , _ = sess.run([model.pred , model.loss, model.train_op] , feed_dict=feed)

		epoch_pred.append(class_pred)
		epoch_label.append(y)
		epoch_loss.append(batch_loss)

		step += 1
		if verbose and not val:
			sys.stdout.write('\r{} / {} :    loss = {}'.format(step*model.config.batch_size, 
																		len(model.X_train), np.mean(epoch_loss)))
			sys.stdout.flush()


	predictions = np.concatenate(epoch_pred , axis=0)
	labels = np.concatenate(epoch_label , axis=0)
	acc = accuracy(labels , predictions)

	return epoch_loss, acc

def train_model(model):

	train_num_batches = int(len(model.X_train)/model.config.batch_size)
	train_loss_history = np.zeros((model.config.max_epochs , train_num_batches)) ## Store each batch separately.
	
	model.config.val_batchsize = model.config.batch_size ## Can be anything, typically greater than train batch size.
	val_num_batches = int(len(model.X_val)/model.config.val_batchsize)
	val_loss_history = np.zeros((model.config.max_epochs , val_num_batches))
	
	train_acc_history = np.zeros((model.config.max_epochs , model.config.label_size)) ## Store each class separately
	val_acc_history = np.zeros_like(train_acc_history) 

	best_val_loss = np.float(np.inf)
	best_epoch = 0

	if not os.path.exists("./weights"):
		os.makedirs("./weights")

	with tf.Session() as sess:

		init = tf.global_variables_initializer()
		sess.run(init)
		# tf.logging.set_verbosity(tf.logging.ERROR)

		for epoch in range(model.config.max_epochs):
			print('Epoch: ' , epoch)
			X_train , seq_len_train , y_train = get_batches(model.X_train , model.y_train, model.config.batch_size)
			epoch_train_loss, epoch_train_acc = run_epoch(sess , model , zip(X_train,seq_len_train,y_train))
			print()
			print("Train Loss: {:.4f} \t Train Accuracy: {} \t Mean Acc: {:.5f}".format(np.mean(epoch_train_loss) ,
																			 epoch_train_acc, np.mean(epoch_train_acc)))


			X_val , seq_len_val , y_val = get_batches(model.X_val , model.y_val, model.config.val_batchsize)
			epoch_val_loss, epoch_val_acc = run_epoch(sess , model , zip(X_val,seq_len_val,y_val) , val=True)
			print("Val Loss: {:.4f} \t Val Accuracy: {} \t Mean Acc: {:.5f}".format(np.mean(epoch_val_loss) , 
																			epoch_val_acc, np.mean(epoch_val_acc)))
			print()

			train_acc_history[epoch , :] = epoch_train_acc
			val_acc_history[epoch , :] = epoch_val_acc
			train_loss_history[epoch , :] = np.array(epoch_train_loss)
			val_loss_history[epoch , :] = np.array(epoch_val_loss)

			val_loss = np.mean(epoch_val_loss)

			if val_loss < best_val_loss:
				best_val_acc = np.mean(epoch_val_acc)
				best_val_loss = val_loss
				best_epoch = epoch
				saver = tf.train.Saver()
				saver.save(sess , './weights/%s'%model.config.model_name)

			if epoch - best_epoch > model.config.anneal_threshold: ## Anneal lr on no improvement in val loss
				model.config.lr *= model.config.annealing_factor
				print("Annealing learning rate to {}".format(model.config.lr))

			if epoch - best_epoch > model.config.early_stopping: ## Stop on no improvement
				print('Stopping due to early stopping')
				break;

	print('Best Validation Accuracy is {}'.format(best_val_acc))

	# plt.plot(np.mean(train_loss_history , axis=0) , linewidth=3 , label='Train')
	# plt.plot(np.mean(val_loss_history , axis=0) , linewidth=3 , label='Val')
	# plt.xlabel("Epoch Number")
	# plt.ylabel("Loss")
	# plt.title("Loss vs Epoch")
	# plt.legend()
	# plt.savefig('Training_graph.png' , format='png')

def test_model(test=False):

	config = Config()
	model = Model(config , 'train.csv' , debug=False)

	start_time = time.time()

	train_model(model) ## Save the weights and model

	print()
	print("#"*20)
	print('Completed Training')
	print('Training Time:{} minutes'.format((time.time()-start_time)/60))

	if not test:
		return

	test_data = pd.read_csv('test.csv')
	X_test = test_data['comment_text'].values
	test_idx = test_data.iloc[:,0].values

	model.config.batch_size = 59*59

	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('./weights/%s.meta'%model.config.model_name)
		saver.restore(sess , './weights/%s'%model.config.model_name)

		X_test , test_seq_length = get_batches(X=X_test, y=None, batch_size=model.config.batch_size , shuffle=False)
		e_pred = []
		for X , seq in zip(X_test , test_seq_length): ## Run test in batches
			feed = model.build_feeddict(X , seq)
			p = sess.run(model.pred , feed_dict=feed)
			e_pred.append(p)

	prediction = np.concatenate(e_pred , axis=0)
	assert(len(test_idx) == len(prediction))

	## Code to write the output submissions to a file

	submit_df = pd.DataFrame({'id':test_idx , 'toxic':prediction[:,0], 'severe_toxic':prediction[:,1], 'obscene':prediction[:,2], 'threat':prediction[:,3], 'insult':prediction[:,4] , 'identity_hate':prediction[:,5]})
	submit_df.to_csv('submission.csv',index=False,columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])


if __name__ == "__main__":

	# config = Config()
	# model_name = input("Enter the module name \n")
	# module = importlib.import_module(name = model_name)
	# Model = getattr(module , 'RNNModel')
	# model = Model(config=config , datafile='train.csv')

	test_model(test=True) ## Load the model and test