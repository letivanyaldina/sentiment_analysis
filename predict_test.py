import os
import sys
import json
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

def predict_unseen_data():
	"""Step 0: LOAD TRAINED MODEL AND PARAMETERS"""
	params = json.loads(open('./parameters.json').read())
	checkpoint_dir = sys.argv[1]
	if not checkpoint_dir.endswith('/'):
		checkpoint_dir += '/'
	checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
	logging.critical('Loaded the trained model: {}'.format(checkpoint_file))
	
	"""Step 1: LOAD DATA FOR PREDICTION"""
	test_file = sys.argv[2]
	test_examples = json.loads(open(test_file).read())

	#labels.json was saved during training, and it has to be loaded during prediction
	labels = json.loads(open('./labels.json').read())
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	x_raw = [example['consumer_complaint_narrative'] for example in test_examples]
	x_test = [data_helper.clean_str(x) for x in x_raw]
	logging.info('The number of x_test: {}'.format(len(x_test)))

	y_test = None
	if 'product' in test_examples[0]:
	y_raw = [example['product'] for example in test_examples]
	y_test = [label_dict[y] for y in y_raw]
	logging.info('The number of y_test: {}'.format(len(y_test)))

	vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
	vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
	x_test = np.array(list(vocab_processor.transform(x_test)))

	"""Step 2: COMPUTE THE PREDICTIONS"""
	
