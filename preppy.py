from collections import defaultdict, Counter
import numpy as np
import tensorflow as tf
PAD = "<PAD>"
START = "<START>"
EOS = "<EOS>"

class Preppy():

	"""
	converts text inputs to numpy arrays of ids.
	It assigns ids sequentially to the token on the fly.
	"""
	def __init__(self, tokenizer_fn):
		self.vocab = defaultdict(self.next_value)
		self.token_counter = Counter()
		self.vocab[PAD] = 0
		self.vocab[START] = 1
		self.vocab[EOS] = 2
		self.next = 2
		self.tokenizer = tokenizer_fn
		self.reverse_vocab = {}

	def next_value(self):
		self.next += 1
		return self.next

	def sequence_to_tf_example(self, sequence):
		"""
		Gets a sequence (a text like "hello how are you") and returns a sequence example.
		:param sequence : some text
		:return: A A sequence example
		"""
		#CONVERT THE TEXT TO A LIST OF IDS
		id_list = self.sentence_to_id_list(sequence)
		ex = tf.train.SequenceExample()
		#A NON-SEQUENTIAL FEATURE OF OUR EXAMPLE
		sequence_length = len(id_list) + 2
		#ADD THE CONTEXT FEATURE, HERE WE JUST NEED LENGTH
		ex.context.feature["length"].int64_list.value.append(sequence_length)
		#FEATURE LISTS FOR THE TWO SEQUENTIAL FEATURES OF OUR EXAMPLE
		#ADD THE TOKENS. THIS IS CORE SEQUENCE
		#YOU CAN ADD ANOTHER SEQUENCE IN THE feature_list dictionary, for translation for instance
		fl_tokens = ex.feature_lists.feature_list["tokens"]
		#PREPEND WITH START TOKEN
		fl_tokens.feature.add().int64_list.value.append(self.vocab[START])
		for token in id_list :
			#ADD THOSE TOKENS ONE BY ONE
			fl_tokens.feature.add().int64_list.value.append(token)
		#APPEND WITH END TOKEN
		fl_tokens.feature.add().int64_list.value.append(self.vocab[EOS])
		return ex

	def ids_to_String(self, tokens, length=None):
		string = ''.join([self.reverse_vocab[x] for x in tokens[:length]])
		return string

	def convert_token_to_id(self, token):
		"""
		Gets a token, look it up in the vocabulary. If it doesn't exist in the vocab, it gets added to id with an id
		Then return the id
		:param token:
		:return: the token id in the vocabulary
		"""
		self.token_counter[token] += 1
		return self.vocab[token]

	def sentence_to_tokens(self, sent):
		return self.tokenizer(sent)

	def tokens_to_id_list(self, tokens):
		return list(map(self.convert_token_to_id, tokens))

	def sentence_to_id_list(self, tokens):
		tokens = self.sentence_to_tokens(sent)
		id_list = self.tokens_to_id_list(tokens)
		return id_list

	def sentence_to_numpy_array(self, sent):
		id_list = self.sentence_to_id_list(sent)
		return np.array(id_list)

	def update_reverse_vocab(self):
		self.reverse_vocab = {id_: token for token, id_ in self.vocab.items()}

	def id_list_to_text(self, id_list):
		tokens = ''.join(map(lambda x: self.reverse_vocab[x], id_list))
		return tokens

	@staticmethod
	def parse(ex):
		"""
		explain to tf how to go from a serialized example back to tensors
		:param ex:
		:return: A dictionary of tensors, in this case {seq: the sequence, length: the length of the sequence}
		"""
		context_features = {
			"length": tf.FixedLenFeature([], dtype=tf.int64)
		}
		sequence_features = {
			"tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
		}

		#PARSE THE EXAMPLE (RETURNS A DICTIONARY OF TENSORS)
		context_parsed, sequence_parsed = tf.parse_single_sequence_example(
			serialized=ex,
			context_features=context_features,
			sequence_features=sequence_features
		)
		return {"seq": sequence_parsed["tokens"],
			"length"}: context_parsed["length"]}

class BibPreppy(Preppy):
	"""
	1. Storing the book_id in the TFRecord
	2. A map from book_ids to book names so we can explore the results
	"""
	
	def __init__(self, tokenizer_fn):
		super(BibPreppy, self).__init__(tokenizer_fn)
		self.book_map = {}
	
	def sequence_to_tf_example(self, sequence, book_id):
		id_list = self.sentence_to_id_list(sequence)
		ex = tf.train.SequenceExample()
		sequence_length = len(sequence)
		ex.context.feature["length"].int64_list.value.append(sequence_length + 2)
		ex.context.feature["book_id"].int64_list.value.append(book_id)

		#feature lists for the two sequential features of our example
		fl_tokens = ex.feature_lists.feature_list["tokens"]
		fl_tokens.feature.add().int64_list.value.append(self.vocab[START])
		for token in id_list:
			fl_tokens.feature.add().int64_list.value.append(token)
		fl_tokens.feature.add().int64_list.value.append(self.vocab[EOS])
		return ex
