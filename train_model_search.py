# 2 functions to search (MLP & ResNet) models over 8 input types of the sequences 
# DC: direct chracters without any converting to kmers
# SK: sequence of kmers without Word2Vec
# W2V: sequence of kmers with Word2Vec

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
from keras.utils.training_utils import multi_gpu_model
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

def search_ResNet_models(input_types,train,valid,database,embedding_matrix,max_len,kmer_size,
	classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
	batch_size = 5
	for x in range(len(input_types)):
		
		if input_types[x] == 'ResNet_SK':

			#ResNet SK
			model = build_resnet(True,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			#model.load_weights(database+'/models/ResNet_SK.hdfs')
			filepath = ''.join(database+'/models/ResNet_SK.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/log/ResNet_SK.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
			train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=50)
			valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=50)
			model.fit_generator(train_generator,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_generator,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])

		elif input_types[x] == 'ResNet_W2V':

			#ResNet W2V
			model = build_resnet(False,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			#model.load_weights(database+'/models/ResNet_W2V.hdfs')
			filepath = ''.join(database+'/models/ResNet_W2V.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/log/ResNet_W2V.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
			train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=50)
			valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=50)
			model.fit_generator(train_generator,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_generator,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
		elif input_types[x] == 'ResNet_SK_fixed_len':
			#ResNet_SK_fixed_len
			model = build_resnet(True,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			filepath = ''.join(database+'/models/ResNet_SK_Fixed_len.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/models/ResNet_SK_Fixed_len.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

			train_gen = simulate_ngs_generator_fixed_len(train,batch_size=batch_size,max_len=max_len,len1=100)
			valid_gen = simulate_ngs_generator_fixed_len(valid,batch_size=batch_size,max_len=max_len,len1=100)

			model.fit_generator(train_gen,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_gen,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
		elif input_types[x] == 'ResNet_W2V_fixed_len':
			#ResNet_W2V_fixed_len
			model = build_resnet(False,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			filepath = ''.join(database+'/models/ResNet_SK_Fixed_len.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/models/ResNet_SK_Fixed_len.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

			train_gen = simulate_ngs_generator_fixed_len(train,batch_size=batch_size,max_len=max_len,len1=100)
			valid_gen = simulate_ngs_generator_fixed_len(valid,batch_size=batch_size,max_len=max_len,len1=100)

			model.fit_generator(train_gen,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_gen,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
		elif input_types[x] == 'ResNet_DC_W2V':
			#ResNet_DC_W2V
			valid = pd.read_pickle(database+'/valid.pkl')
			train = pd.read_pickle(database+'/train.pkl')
			train = train.sample(frac=1).reset_index(drop=True)
			valid = valid.sample(frac=1).reset_index(drop=True)
			train = train.drop(columns=['Complete_species','ambiguity_count'])
			valid = valid.drop(columns=['Complete_species','ambiguity_count'])

			model = build_resnet(False,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			filepath = ''.join(database+'/models/ResNet_DC_W2V.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/log/ResNet_DC_W2V.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
			train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=50)
			valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=50)
			model.fit_generator(train_generator,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_generator,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
		else:
			#ResNet_DC_No_W2V
			valid = pd.read_pickle(database+'/valid.pkl')
			train = pd.read_pickle(database+'/train.pkl')
			train = train.sample(frac=1).reset_index(drop=True)
			valid = valid.sample(frac=1).reset_index(drop=True)
			train = train.drop(columns=['Complete_species','ambiguity_count'])
			valid = valid.drop(columns=['Complete_species','ambiguity_count'])

			model = build_resnet(True,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			filepath = ''.join(database+'/models/ResNet_DC_No_W2V.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/log/ResNet_DC_No_W2V.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
			train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=50)
			valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=50)
			model.fit_generator(train_generator,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_generator,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])


def search_MLP_models(input_types,train,valid,database,embedding_matrix,max_len,kmer_size,
	classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):

	batch_size = 5

	for x in range(len(input_types)):

		if input_types[x] == 'MLP_SK':

			#MLP SK
			model = build_mlp(True,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			filepath = ''.join(database+'/models/MLP_SK.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/log/MLP_SK.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
			train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=50)
			valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=50)
			model.fit_generator(train_generator,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_generator,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])

		elif input_types[x] == 'MLP_W2V':

			#MLP W2V
			batch_size = 5
			model = build_mlp(False,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			filepath = ''.join(database+'/models/MLP_W2V.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/log/MLP_W2V.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
			train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=50)
			valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=50)
			model.fit_generator(train_generator,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_generator,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
		elif input_types[x] == 'MLP_SK_fixed_len':
			#MLP_SK_fixed_len
			model = build_mlp(True,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
			#model.load_weights(database+'/models/Fixed_len_MLP_like_N.hdfs')
			filepath = ''.join(database+'/models/MLP_SK_Fixed_len.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/models/MLP_SK_Fixed_len.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

			train_gen = simulate_ngs_generator_fixed_len(train,batch_size=batch_size,max_len=max_len,len1=100)
			valid_gen = simulate_ngs_generator_fixed_len(valid,batch_size=batch_size,max_len=max_len,len1=100)

			model.fit_generator(train_gen,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_gen,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
		elif input_types[x] == 'MLP_W2V_fixed_len':
			#MLP_W2V_fixed_len
			model = build_mlp(False,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
			#model.load_weights(database+'/models/Fixed_len_MLP_like_N.hdfs')
			filepath = ''.join(database+'/models/MLP_SK_Fixed_len.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/models/MLP_SK_Fixed_len.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

			train_gen = simulate_ngs_generator_fixed_len(train,batch_size=batch_size,max_len=max_len,len1=100)
			valid_gen = simulate_ngs_generator_fixed_len(valid,batch_size=batch_size,max_len=max_len,len1=100)

			model.fit_generator(train_gen,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_gen,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
		elif input_types[x] == 'MLP_DC_W2V':
			#MLP_DC_W2V
			valid = pd.read_pickle(database+'/valid.pkl')
			train = pd.read_pickle(database+'/train.pkl')
			train = train.sample(frac=1).reset_index(drop=True)
			valid = valid.sample(frac=1).reset_index(drop=True)
			train = train.drop(columns=['Complete_species','ambiguity_count'])
			valid = valid.drop(columns=['Complete_species','ambiguity_count'])

			model = build_mlp(False,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			filepath = ''.join(database+'/models/MLP_DC_W2V.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/log/MLP_DC_W2V.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
			train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=50)
			valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=50)
			model.fit_generator(train_generator,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_generator,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
		else:
			#MLP_DC_No_W2V
			valid = pd.read_pickle(database+'/valid.pkl')
			train = pd.read_pickle(database+'/train.pkl')
			train = train.sample(frac=1).reset_index(drop=True)
			valid = valid.sample(frac=1).reset_index(drop=True)
			train = train.drop(columns=['Complete_species','ambiguity_count'])
			valid = valid.drop(columns=['Complete_species','ambiguity_count'])

			model = build_mlp(True,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
			filepath = ''.join(database+'/models/MLP_DC_No_W2V.hdfs')
			Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
			csv_logger = CSVLogger(database+'/log/MLP_DC_No_W2V.log')
			EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
			train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=50)
			valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=50)
			model.fit_generator(train_generator,
				                epochs=50,shuffle=True,
				                steps_per_epoch=train.shape[0]//batch_size,
				                validation_data=valid_generator,
				                validation_steps=valid.shape[0]//batch_size,
				                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])