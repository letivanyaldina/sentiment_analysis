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

def load_data_and_labels(filename):

	df = pd.read_csv(filename, compression='zip', dtype={'content':object})
	
	selected = ['final_sentiment','content']
	non_selected = list(set(df.columns) - set(selected))

	
	"""drop columns other than final_sentiment and content"""
	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how='any', subset=selected)
	df = df.reindex(np.random.permutation(df.index))
	"""drop rows other than 0, 1, 2"""
	df = df[df.final_sentiment != -1]
	#test
	#print (df)
	#df = df[df.final_sentiment == 1]
	#df = df[df.final_sentiment == 2]
	
	
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
	input_file = 'data_nissan.csv.zip'
	load_data_and_labels(input_file)


	


