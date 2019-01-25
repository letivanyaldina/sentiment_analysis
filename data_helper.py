""" file for data processing"""

import re
import logging
import numpy as np
import pandas as pd
from collections import Counter

def clean_str(s):
	"""
	string yang di-replace apa aja
	string yang nge-replace
	
	"""
	s = re.sub(r"[^A-Za-z0-9(), !?\'\`]", " ", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ",s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	s = re.sub(r'\S*(x{2,}|X{2,})\S*', "xxx", s)
	s = re.sub(r'[^\x00-\x7F]+', "", s)
	s = re.sub(r"\'", " ", s)
	s = re.sub(r"\\", " ", s)
	return s.strip().lower()

"""
string_try = clean_str("'1364734936996607_1365466750256759,01/11/2018 22:24,'social_media,'facebook,'comment,'account,Datsun.indonesia ,,,,'Nah yang begini nih yang cocok buat anak muda jaman now..,2,'https://graph.facebook.com//picture,[],'https://facebook.com/1364734936996607_1365466750256759,,'275216562615122_1364734936996607,'275216562615122,,,[],0,0,0,0,,,,[],0,,,,,")

print(string_try)

"""

def load_data_and_labels(pos_filename, neg_filename, neu_filename):

	#df = pd.read_csv(filename, compression='zip', dtype={'content':object})
	positive_examples = pd.read_csv(pos_filename, compression='zip', dtype={'content':object})
	negative_examples = pd.read_csv(neg_filename, compression='zip', dtype={'content':object})
	neutral_examples = pd.read_csv(neu_filename, compression='zip', dtype={'content':object})
	
	
	#proses satu2 untuk masing2 pos, neg, neu
	#1. pilih kolom final_sentiment & content
	#2. drop kolom yg lain
	#3. baru gabungkan df masing2
	
	"""positive data processing"""
	df_pos = positive_examples
	pos_selected = ['final_sentiment', 'content']
	pos_nonselected = list(set(df_pos.columns) - set(pos_selected))
	#drop columns
	df_pos = df_pos.drop(pos_nonselected, axis=1)
	df_pos = df_pos.dropna(axis=0, how='any', subset=pos_selected)
	df_pos = df_pos.reindex(np.random.permutation(df_pos.index))
	#drop rows
	df_pos = df_pos[df_pos.final_sentiment != -1]


	"""negative data processing"""
		
	df_neg = negative_examples
	neg_selected = ['final_sentiment', 'content']
	neg_nonselected = list(set(df_neg.columns) - set(neg_selected))
	#drop columns
	df_neg = df_neg.drop(neg_nonselected, axis=1)
	df_neg = df_neg.dropna(axis=0, how='any', subset=neg_selected)
	df_neg = df_neg.reindex(np.random.permutation(df_neg.index))
	#drop rows
	df_neg = df_neg[df_neg.final_sentiment != -1]
	print (df_neg)

	"""neutral data processing"""
	df_neu = neutral_examples
	neu_selected = ['final_sentiment', 'content']
	neu_nonselected = list(set(df_neu.columns) - set(neu_selected))
	#drop columns
	df_neu = df_neu.drop(neu_nonselected, axis=1)
	df_neu = df_neu.dropna(axis=0, how='any', subset=neu_selected)
	df_neu = df_neu.reindex(np.random.permutation(df_neu.index))
	#drop rows
	df_neu = df_neu[df_neu.final_sentiment != -1]
	
	
	df = pd.concat([df_pos, df_neg, df_neu])
	selected = ['final_sentiment', 'content']
	print (df)
	
	labels = sorted(list(set(df[selected[0]].tolist())))
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
	return x_raw, y_raw, df, labels


"""
filename = load_data_and_labels('data_nissan.csv.zip')

print(filename)
"""




def batch_iter(data, batch_size, num_epochs, shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs) :
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
	
	for batch_num in range(num_batches_per_epoch):
		start_index = batch_num * batch_size
		end_index = min((batch_num+1) * batch_size, data_size)
		yield shuffled_data[start_index:end_index]





if __name__ == '__main__':
	#input_file = 'data_nissan.csv.zip'
	pos_filename = 'positive.csv.zip'
	neg_filename = 'negative.csv.zip'
	neu_filename = 'neutral.csv.zip'
	#load_data_and_labels(input_file)
	load_data_and_labels(pos_filename, neg_filename, neu_filename)


	


