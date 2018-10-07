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
from gensim.models import KeyedVectors
import os
import sys
from train_func import *
from train_model_search import *

lengths=[25,50,75,100,125,'full_HVR']
batch_size = 5
def evaluate_multioutput_on_different_length(model, lengths,data,batch_size,save_path,max_len):
    eval_df =  pd.DataFrame(np.zeros((6,len(lengths))),columns=lengths,index=['Phylum','Order','Class','Family','Genus','Species'])
    for len_ in range(len(lengths[:-1])):
        gen = simulate_ngs_generator_fixed_len(data,len1=lengths[len_],batch_size=batch_size,max_len=max_len)
        eval_df.iloc[:,len_] = model.evaluate_generator(gen,verbose=2,steps=data.shape[0]//batch_size)[-6:]
        eval_df.to_csv(save_path)  
    data['simulated'] = pad_sequences(data['encoded'].values,maxlen=max_len).tolist()
    y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5,y_sim_6 = data['phylum-'].values,data['class_-'].values,data['order-'].values,data['family-'].values,data['genus-'].values,data['species-'].values
    x_sim = data['simulated']
    x_sim = array(np.concatenate(x_sim.values).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
    eval_df.iloc[:,-1]= model.evaluate(x_sim,[y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5,y_sim_6 ],batch_size=50)[-6:]
    eval_df.to_csv(save_path)
    return eval_df


def evaluate_ResNet_models(input_types,test,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    test  = pd.read_pickle(database+'/test.pkl')
    test  = test.sample(frac=1).reset_index(drop=True)
    test  = test.drop(columns=['Complete_species','ambiguity_count'])
    test['encoded'] = test['encoded'].apply(lambda  x: oneHotEncoding_to_kmers(x,kmer_size=kmer_size)).values.tolist()

    for x in range(len(input_types)):

        if input_types[x] == 'ResNet_SK':

            #ResNet SK
            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_SK.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_SK.csv'),max_len=max_len)

        elif input_types[x] == 'ResNet_W2V':

            #ResNet W2V
            model = build_resnet(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_W2V.csv'),max_len=max_len)
        elif input_types[x] == 'ResNet_SK_fixed_len':
            #ResNet_SK_fixed_len
            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_SK_Fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_SK_Fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'ResNet_W2V_fixed_len':
            #ResNet_W2V_fixed_len
            model = build_resnet(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_W2V_fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_W2V_fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'ResNet_DC_W2V':
            #ResNet_DC_W2V
            test  = pd.read_pickle(database+'/test.pkl')
            test  = test.sample(frac=1).reset_index(drop=True)
            test  = test.drop(columns=['Complete_species','ambiguity_count'])

            model = build_resnet(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_DC_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_DC_W2V.csv'),max_len=max_len)
        else:
            #ResNet_DC_No_W2V
            test  = pd.read_pickle(database+'/test.pkl')
            test  = test.sample(frac=1).reset_index(drop=True)
            test  = test.drop(columns=['Complete_species','ambiguity_count'])

            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_DC_No_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_DC_No_W2V.csv'),max_len=max_len)


def evaluate_MLP_models(input_types,test,database,embedding_matrix,max_len,kmer_size,metrics,metrics,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    test  = pd.read_pickle(database+'/test.pkl')
    test  = test.sample(frac=1).reset_index(drop=True)
    test  = test.drop(columns=['Complete_species','ambiguity_count'])
    test['encoded'] = test['encoded'].apply(lambda  x: oneHotEncoding_to_kmers(x,kmer_size=kmer_size)).values.tolist()

    for x in range(len(input_types)):

        if input_types[x] == 'MLP_SK':

            #MLP SK
            model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_SK.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_SK.csv'),max_len=max_len)
        elif input_types[x] == 'MLP_W2V':

            #MLP W2V
            model = build_mlp(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_W2V.csv'),max_len=max_len)

        elif input_types[x] == 'MLP_SK_fixed_len':
            #MLP_SK_fixed_len
            model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_SK_fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_SK_fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'MLP_W2V_fixed_len':
            #MLP_W2V_fixed_len
            model = build_mlp(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_W2V_fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_W2V_fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'MLP_DC_W2V':
            #MLP_DC_W2V
            test  = pd.read_pickle(database+'/test.pkl')
            test  = test.sample(frac=1).reset_index(drop=True)
            test  = test.drop(columns=['Complete_species','ambiguity_count'])

            model = build_mlp(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_DC_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_DC_W2V.csv'),max_len=max_len)
        else:
            #MLP_DC_No_W2V
            test  = pd.read_pickle(database+'/test.pkl')
            test  = test.sample(frac=1).reset_index(drop=True)
            test  = test.drop(columns=['Complete_species','ambiguity_count'])

            model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_DC_No_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_DC_No_W2V.csv'),max_len=max_len)


def evaluate_best_only(database,embedding_matrix,max_len,kmer_size,metrics,test,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    test  = pd.read_pickle(database+'/test.pkl')
    test  = test.sample(frac=1).reset_index(drop=True)
    test  = test.drop(columns=['Complete_species','ambiguity_count'])
    test['encoded'] = test['encoded'].apply(lambda  x: oneHotEncoding_to_kmers(x,kmer_size=kmer_size)).values.tolist()

    #MLP SK
    model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.load_weights(database+'/models/MLP_SK.hdfs')
    evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                             batch_size=50,save_path=''.join(database+'/results/MLP_SK.csv'),max_len=max_len)