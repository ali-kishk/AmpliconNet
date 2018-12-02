from __future__ import print_function
import numpy as np
import pandas as pd 
from numpy import array
from numpy.random import randint

import random

from sklearn.metrics import r2_score, accuracy_score

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Concatenate, LeakyReLU, concatenate,GRU, Bidirectional, MaxPool1D,GlobalMaxPool1D,add
from keras.layers import Dense, Embedding, Input, Masking, Dropout, MaxPooling1D,Lambda, BatchNormalization, Reshape
from keras.layers import LSTM, TimeDistributed, AveragePooling1D, Flatten,Activation,ZeroPadding1D
from keras.optimizers import Adam, rmsprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint, CSVLogger
from keras.layers import Conv1D, GlobalMaxPooling1D, ConvLSTM2D, Bidirectional,RepeatVector
from keras.regularizers import *
from keras import regularizers
from keras.layers import concatenate as concatLayer
from keras.utils import plot_model

import itertools
from itertools import product

from gensim.models import KeyedVectors
import os
import sys
from train_func import *
from train_model_search import *

import argparse

parser = argparse.ArgumentParser(description='SpeciesMLP: 16S rRNA taxonomic classifier using deep learning')
parser.add_argument('--database_dir', dest='database_dir',type=str, default='.', help='Input directory contains train, test & valid data')
parser.add_argument('--kmer_size', dest='kmer_size', type=int, default=6, help='kmer size to convert the sequence of reads to sequence of kmers')
parser.add_argument('--max_len', dest='max_len', type=int, default=320, help='A maximum length of all reads in a multifasta database \\\
	for zero padding, You should increase it more than the actual maximum length if you are expecting longer reads in the prediction')
parser.add_argument('--training_mode', dest='training_mode',type=str, default='best_only', help='Training search mode :\\\
	\n - search_mlp : to search the multiperceptron model, \\\
	\n - Search_resnet : to search teh ResNet model \\\
	\n - best_only : to train only our best model ( MLP over sequence of kmers without word2vec) ')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=250, help='Training batch size, default 250')
parser.add_argument('--load_mode', dest='load_mode', type=bool, default=False, help='If the model should load pretrained model, default False')
parser.add_argument('--metrics', dest='metrics', type=str, default='accuracy', help='accuracy or f1 training metrics')
parser.add_argument('--genus_only', dest='genus_only', type=bool, default=True, help='Whetether to model the genus level onlyOR to  add species level')

# Parameters
args = parser.parse_args()

database = args.database_dir
kmer_size = args.kmer_size
batch_size = args.batch_size
max_len = args.max_len
search = args.training_mode #best_only, search_resnet, search_mlp
load_mode = args.load_mode
metrics = args.metrics
genus_only = args.genus_only

bases=['1','2','3','4']
all_kmers = [''.join(p) for p in itertools.product(bases, repeat=kmer_size)]
word_to_int = dict()
word_to_int = word_to_int.fromkeys(all_kmers)
keys = range(1,len(all_kmers)+1)
for k in keys:
    word_to_int[all_kmers[k-1]] = keys[k-1]


def main():
	if search != 'best_only':
		embedding_matrix = build_embedding_matrix(database+'/W2V_model_'+str(kmer_size)+'_kmer.w2v', word_to_int)
	else:
		embedding_matrix = 0

	valid = pd.read_pickle(database+'/valid.pkl')
	train = pd.read_pickle(database+'/train.pkl')


	train['len'] = train['encoded'].apply(lambda x: len(x))
	train = train[train['len']>50]
	valid['len'] = valid['encoded'].apply(lambda x: len(x))
	valid = valid[valid['len']>50]
	
	train = train.sample(frac=1).reset_index(drop=True)
	valid = valid.sample(frac=1).reset_index(drop=True)

	max_features = 4**kmer_size +1
	vector_size = 128

	train['encoded'] = train['encoded'].apply(lambda x: oneHotEncoding_to_kmers(x,kmer_size=kmer_size)).values.tolist()
	valid['encoded'] = valid['encoded'].apply(lambda x: oneHotEncoding_to_kmers(x,kmer_size=kmer_size)).values.tolist()


	classes_1 = max(train['phylum'])  +1
	classes_2 = max(train['class_'])  +1
	classes_3 = max(train['order'])   +1
	classes_4 = max(train['family'])  +1
	classes_5 = max(train['genus'])   +1
	classes_6 = 0
	
	if genus_only == True:
		classes_6 = 0
	else:
		classes_6 = max(train['species']) +1

	if 'models' not in os.listdir(database) :
		if load_mode == False:
			os.mkdir(database+'/models')
			os.mkdir(database+'/log')
			os.mkdir(database+'/results')
		else:
			pass
	else:
		pass
# DC: direct chracters without any converting to kmers
# SK: sequence of kmers without Word2Vec
# W2V: sequence of kmers with Word2Vec

	all_resnet_models = ['ResNet_SK','ResNet_W2V','ResNet_SK_fixed_len','ResNet_W2V_fixed_len','ResNet_DC_W2V','ResNet_DC_No_W2V']
	all_mlp_models = ['MLP_SK','MLP_W2V','MLP_SK_fixed_len','MLP_W2V_fixed_len','MLP_DC_W2V','MLP_DC_No_W2V']
	all_gru_models = ['GRU_SK','GRU_W2V','GRU_SK_fixed_len','GRU_W2V_fixed_len','GRU_DC_W2V','GRU_DC_No_W2V']

	if search == 'search_mlp':
		search_MLP_models(all_mlp_models,train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
			classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)

	elif search == 'search_resnet':
		search_ResNet_models(all_resnet_models,train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
			classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)

	elif search == 'search_gru':
		search_GRU_models(all_gru_models,train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
			classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)

	elif search == 'mlp_sk':
		train_mlp_only(train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,load_mode,
			classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)

	elif search == 'resnet_sk':
		train_resnet_only('ResNet_SK',train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
			classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)

	elif search == 'gru_sk':
		train_gru_only(train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,load_mode,
			classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)

	elif search == 'multi_task':
		multi_task_training(train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,load_mode,
			classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
	else:
		print('Wrong search type ! ')

if __name__ == "__main__":
    main()
