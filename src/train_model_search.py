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
import tensorflow as tf
from tensorflow.python.keras.layers import Input, GlobalMaxPool1D, Dense, Embedding,Dropout

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
from keras.utils import plot_model, to_categorical

import itertools
from itertools import product

from gensim.models import KeyedVectors
import os
import sys
from train_func import *

#f1_on_all = f1_on_all()

def search_ResNet_models(input_types,train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):

    for x in range(len(input_types)):
        
        if input_types[x] == 'ResNet_DC_No_W2V':
            #ResNet_DC_No_W2V
            valid = pd.read_pickle(database+'/valid.pkl')
            train = pd.read_pickle(database+'/train.pkl')
            train = train.sample(frac=1).reset_index(drop=True)
            valid = valid.sample(frac=1).reset_index(drop=True)

            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/ResNet_DC_No_W2V.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/ResNet_DC_No_W2V.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])


        elif input_types[x] == 'ResNet_W2V':

            #ResNet W2V
            model = build_resnet(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            #model.load_weights(database+'/models/ResNet_W2V.hdfs')
            filepath = ''.join(database+'/models/ResNet_W2V.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/ResNet_W2V.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])

        elif input_types[x] == 'ResNet_SK_fixed_len':
            #ResNet_SK_fixed_len
            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
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
            model = build_resnet(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
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

            model = build_resnet(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/ResNet_DC_W2V.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/ResNet_DC_W2V.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
        else:

            #ResNet SK
            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/ResNet_SK.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/ResNet_SK.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])

def search_MLP_models(input_types,train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):

    for x in range(len(input_types)):

        if input_types[x] == 'MLP_DC_No_W2V':


            #MLP_DC_No_W2V
            valid = pd.read_pickle(database+'/valid.pkl')
            train = pd.read_pickle(database+'/train.pkl')
            train = train.sample(frac=1).reset_index(drop=True)
            valid = valid.sample(frac=1).reset_index(drop=True)

            model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/MLP_DC_No_W2V.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/MLP_DC_No_W2V.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
        elif input_types[x] == 'MLP_W2V':

            #MLP W2V
            model = build_mlp(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/MLP_W2V.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/MLP_W2V.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
        elif input_types[x] == 'MLP_SK_fixed_len':
            #MLP_SK_fixed_len
            model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
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
            model = build_mlp(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
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

            model = build_mlp(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/MLP_DC_W2V.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/MLP_DC_W2V.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
        else:
            #MLP SK
            model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/MLP_SK.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/MLP_SK.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])


def search_GRU_models(input_types,train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):

    for x in range(len(input_types)):

        if input_types[x] == 'GRU_DC_No_W2V':

            #GRU_DC_No_W2V
            valid = pd.read_pickle(database+'/valid.pkl')
            train = pd.read_pickle(database+'/train.pkl')
            train = train.sample(frac=1).reset_index(drop=True)
            valid = valid.sample(frac=1).reset_index(drop=True)

            model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/GRU_DC_No_W2V.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/GRU_DC_No_W2V.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
        elif input_types[x] == 'GRU_W2V':

            #GRU W2V
            model = build_gru(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/GRU_W2V.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/GRU_W2V.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
        elif input_types[x] == 'GRU_SK_fixed_len':
            #GRU_SK_fixed_len
            model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
            filepath = ''.join(database+'/models/GRU_SK_Fixed_len.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/models/GRU_SK_Fixed_len.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

            train_gen = simulate_ngs_generator_fixed_len(train,batch_size=batch_size,max_len=max_len,len1=100)
            valid_gen = simulate_ngs_generator_fixed_len(valid,batch_size=batch_size,max_len=max_len,len1=100)

            model.fit_generator(train_gen,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_gen,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
        elif input_types[x] == 'GRU_W2V_fixed_len':
            #GRU_W2V_fixed_len
            model = build_gru(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
            filepath = ''.join(database+'/models/GRU_SK_Fixed_len.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/models/GRU_SK_Fixed_len.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

            train_gen = simulate_ngs_generator_fixed_len(train,batch_size=batch_size,max_len=max_len,len1=100)
            valid_gen = simulate_ngs_generator_fixed_len(valid,batch_size=batch_size,max_len=max_len,len1=100)

            model.fit_generator(train_gen,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_gen,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
        elif input_types[x] == 'GRU_DC_W2V':
            #GRU_DC_W2V
            valid = pd.read_pickle(database+'/valid.pkl')
            train = pd.read_pickle(database+'/train.pkl')
            train = train.sample(frac=1).reset_index(drop=True)
            valid = valid.sample(frac=1).reset_index(drop=True)

            model = build_gru(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/GRU_DC_W2V.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/GRU_DC_W2V.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])
        else:

            #GRU SK
            model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            filepath = ''.join(database+'/models/GRU_SK.hdfs')
            Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
            csv_logger = CSVLogger(database+'/log/GRU_SK.log')
            EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
            model.fit_generator(train_generator,
                                epochs=50,shuffle=True,
                                steps_per_epoch=train.shape[0]//batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid.shape[0]//batch_size,
                                verbose=2,callbacks=[Checkpoint,csv_logger,EarlyStop])



def train_mlp_only(train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,load_mode,tpu,save_mem,min_len,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    #MLP SK
    if tpu ==True:
        model = build_tpu_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    else:
        model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)

    if load_mode ==True:
        model.load_weights(database+'/models/MLP_SK.hdfs')
    else:
        pass
    filepath = ''.join(database+'/models/MLP_SK.hdfs')
    Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
    csv_logger = CSVLogger(database+'/log/MLP_SK.log')

    EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    vector_size = 128
    train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
    valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
    if save_mem ==True:
        steps_per_epoch= pd.read_csv(train).shape[0]//batch_size #sum(1 for line in open(train))//batch_size
        validation_steps=pd.read_csv(valid).shape[0]//batch_size #sum(1 for line in open(valid))//batch_size
    else:
        steps_per_epoch=train.shape[0]//batch_size
        validation_steps=valid.shape[0]//batch_size
    model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,
                        epochs=50,shuffle=True,
                        validation_data=valid_generator,
                        validation_steps=validation_steps,
                        verbose=1,
                        callbacks=[Checkpoint,csv_logger,EarlyStop])

def train_gru_only(train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,load_mode,tpu,save_mem,min_len,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    #MLP SK
    model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    filepath = ''.join(database+'/models/GRU_SK.hdfs')
    Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
    csv_logger = CSVLogger(database+'/log/GRU_SK.log')
    if load_mode ==True:
        model.load_weights(database+'/models/GRU_SK.hdfs')
    else:
        pass
    EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
    valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
    if save_mem ==True:
        steps_per_epoch= pd.read_csv(train).shape[0]//batch_size #sum(1 for line in open(train))//batch_size
        validation_steps=pd.read_csv(valid).shape[0]//batch_size #sum(1 for line in open(valid))//batch_size
    else:
        steps_per_epoch=train.shape[0]//batch_size
        validation_steps=valid.shape[0]//batch_size
    model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,
                        epochs=50,shuffle=True,
                        validation_data=valid_generator,
                        validation_steps=validation_steps,
                        verbose=1,
                        callbacks=[Checkpoint,csv_logger,EarlyStop])

def train_resnet_only(train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,load_mode,tpu,save_mem,min_len,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    #MLP SK
    model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    filepath = ''.join(database+'/models/ResNet_SK.hdfs')
    Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
    csv_logger = CSVLogger(database+'/log/ResNet_SK.log')
    if load_mode ==True:
        model.load_weights(database+'/models/ResNet_SK.hdfs')
    else:
        pass
    EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    train_generator = simulate_ngs_generator(train,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
    valid_generator = simulate_ngs_generator(valid,batch_size=batch_size,max_len=max_len,len1=min_len,save_mem=save_mem,kmer_size=kmer_size)
    if save_mem ==True:
        steps_per_epoch= pd.read_csv(train).shape[0]//batch_size #sum(1 for line in open(train))//batch_size
        validation_steps=pd.read_csv(valid).shape[0]//batch_size #sum(1 for line in open(valid))//batch_size
    else:
        steps_per_epoch=train.shape[0]//batch_size
        validation_steps=valid.shape[0]//batch_size
    model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,
                        epochs=50,shuffle=True,
                        validation_data=valid_generator,
                        validation_steps=validation_steps,
                        verbose=1,
                        callbacks=[Checkpoint,csv_logger,EarlyStop])

def multi_task_training(train,valid,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,load_mode,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    #MLP SK
    model = build_multitask_mlp(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    filepath = ''.join(database+'/models/Multitask_MLP_SK.hdfs')
    Checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,save_weights_only=True,period=1)
    csv_logger = CSVLogger(database+'/log/MLP_SK.log')
    if load_mode ==True:
        model.load_weights(database+'/models/Multitask_MLP_SK.hdfs')
    else:
        pass
    EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    train_generator = simulate_multi_task_generator(train,50,batch_size,max_len,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    valid_generator = simulate_multi_task_generator(valid,50,batch_size,max_len,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    #RocAuc = RocAucEvaluation((valid),1,batch_size,max_len,
    #classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.fit_generator(train_generator,steps_per_epoch=train.shape[0]//batch_size,
                        epochs=50,shuffle=True,
                        validation_data=valid_generator,
                        validation_steps=valid.shape[0]//batch_size,
                        verbose=1,
                        callbacks=[Checkpoint,csv_logger,EarlyStop])

# AUC callback for multitasking
"""
from sklearn.metrics import roc_auc_score,jaccard_similarity_score
class RocAucEvaluation(Callback):
    def __init__(self, df=(), interval=1,batch_size = 250,max_len = 321,
                classes_1 = 25,classes_2 = 50,classes_3 = 100,classes_4 = 500,classes_5 = 1000,classes_6 = 10000):
        super(Callback, self).__init__()

        self.interval = interval
        self.df = df
        self.batch_size= batch_size
        self.max_len = max_len
        self.classes_1,self.classes_2,self.classes_3 = classes_1,classes_2,classes_3 
        self.classes_4,self.classes_5,self.classes_6 = classes_4,classes_5,classes_6

#     def on_batch_end(self, batch, logs={}):
#         y_pred = self.model.predict(self.X_val, verbose=0)
#         score = roc_auc_score(self.y_val, y_pred)
#         jac_score = jaccard_similarity_score(self.y_val, y_pred.round(), normalize=False)
#         logs['roc_auc_val'] = score
#         logs['jac_val'] = jac_score
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_1,y_2,y_3 = self.df['phylum-'].values, self.df['class_-'].values, self.df['order-'].values
            y_4,y_5,y_6 = self.df['family-'].values, self.df['genus-'].values, self.df['species-'].values
            y_1 = to_categorical(y_1,num_classes=self.classes_1)
            y_2 = to_categorical(y_2,num_classes=self.classes_2)
            y_3 = to_categorical(y_3,num_classes=self.classes_3)
            y_4 = to_categorical(y_4,num_classes=self.classes_4)
            y_5 = to_categorical(y_5,num_classes=self.classes_5)
            y_6 = to_categorical(y_6,num_classes=self.classes_6)
            y_val = np.hstack((y_1,y_2,y_3,y_4,y_5,y_6))
            val_gen = simulate_multi_task_generator(self.df,50,self.batch_size,self.max_len,
                self.classes_1,self.classes_2,self.classes_3,self.classes_4,self.classes_5,self.classes_6)
            y_pred = self.model.predict_generator(val_gen,steps=self.df.shape[0]//self.batch_size).argmax(axis=-1)
            y_pred = y_pred.argmax(axis=-1)
            score = roc_auc_score(y_val, y_pred)
            jac_score = jaccard_similarity_score(y_val, y_pred.round())
            logs['roc_auc_val'] = score
            logs['jac_val'] = jac_score
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            print("\n JAC-SIM - epoch: %d - score: %.6f \n" % (epoch+1, jac_score))"""
