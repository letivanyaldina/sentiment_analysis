import os
import sys
import json
import time
import logging
import data_helper
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

def train_cnn() :
	"""Step 0: LOAD SENTENCES, LABELS, AND TRAINING PARAMETERS"""
	train_file = sys.argv[1]
	#test
	#print (train_file)
	
	x_raw, y_raw, df, labels = data_helper.load_data_and_labels(train_file)
	#test
	#print (x_raw)
	#print (y_raw)
	#print (df)
	#print (labels)
	parameter_file = sys.argv[2]
	#test
	#print (parameter_file)
	params = json.loads(open(parameter_file).read())
	#test
	#print (params)
	
	"""Step 1: PAD EACH SENTENCE TO THE SAME LENGTH AND MAP EACH WORD TO AN ID"""
	max_document_length = max([len(x.split(' ')) for x in x_raw])
	#test
	#print (max_document_length)
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	#test
	print (vocab_processor)
	"""x = np.array(list(vocab_processor.fit_transform(x_raw)))
	y = np.array(y_raw)"""

	"""Step 2: SPLIT THE ORIGINAL DATASET INTO TRAIN AND TEST SETS"""
	"""x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)"""

	"""Step 3: SHUFFLE THE TRAIN SET AND SPLIT THE TRAIN SET INTO TRAIN AND DEV SETS
	shuffle_indices = np.random.permutation(np.arange(len(y_)))
	x_shuffled = x_[shuffle_indices]
	y_shuffled = y_[shuffle_indices]
	x_train, x_dev, y_train
"""

if __name__ == '__main__':
	train_cnn()
