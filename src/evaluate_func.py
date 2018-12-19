from __future__ import print_function
import numpy as np
import pandas as pd 
from numpy import array
from numpy.random import randint

import random

from sklearn.metrics import r2_score, accuracy_score, roc_auc_score,jaccard_similarity_score
from sklearn.metrics import f1_score,precision_score,recall_score


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
from sklearn.datasets import make_classification


lengths=[25,50,75,100,125,'full_HVR']
batch_size = 5

"""
#Accuracy evaluation on multi-output
def evaluate_multioutput_on_different_length(model, lengths,data,batch_size,save_path,max_len):
    eval_df =  pd.DataFrame(np.zeros((5,len(lengths))),columns=lengths,index=['Phylum','Order','Class','Family','Genus'])
    for len_ in range(len(lengths[:-1])):
        gen = simulate_ngs_generator_fixed_len(data,len1=lengths[len_],batch_size=batch_size,max_len=max_len)
        eval_df.iloc[:,len_] = model.evaluate_generator(gen,verbose=2,steps=data.shape[0]//batch_size)[-5:]
        eval_df.to_csv(save_path)  
    data['simulated'] = pad_sequences(data['encoded'].values,maxlen=max_len).tolist()
    y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = data['phylum'].values,data['class_'].values,data['order'].values,data['family'].values,data['genus'].values
    x_sim = data['simulated']
    x_sim = array(np.concatenate(x_sim.values).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
    eval_df.iloc[:,-1]= model.evaluate(x_sim,[y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 ],batch_size=50)[-5:]
    eval_df.to_csv(save_path)
    return eval_df
"""

# Precision evaluation on multi-output
def evaluate_multioutput_on_different_length(model, lengths,data,batch_size,save_path,max_len,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    #measuring accuracy first
    """eval_df =  pd.DataFrame(np.zeros((5,len(lengths))),columns=lengths,index=['Phylum','Order','Class','Family','Genus'])
    for len_ in range(len(lengths[:-1])):
        gen = simulate_ngs_generator_fixed_len(data,len1=lengths[len_],batch_size=batch_size,max_len=max_len)
        eval_df.iloc[:,len_] = model.evaluate_generator(gen,verbose=2,steps=data.shape[0]//250)[-5:]
        eval_df.to_csv(save_path)  
    data['simulated'] = pad_sequences(data['encoded'].values,maxlen=max_len).tolist()
    y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = data['phylum'].values,data['class_'].values,data['order'].values,data['family'].values,data['genus'].values
    x_sim = data['simulated']
    x_sim = array(np.concatenate(x_sim.values).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
    eval_df.iloc[:,-1]= model.evaluate(x_sim,[y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 ],batch_size=250)[-5:]
    eval_df.to_csv(save_path+'_accuracy.csv')
"""
    num_steps = data.shape[0]//batch_size
    data = data.iloc[:num_steps*batch_size,:]
    metrics = ['f1', 'precision','recall']

    eval_df_1 =  pd.DataFrame(np.zeros((5,len(lengths))),columns=lengths,index=['Phylum','Order','Class','Family','Genus'])
    eval_df_2 =  pd.DataFrame(np.zeros((5,len(lengths))),columns=lengths,index=['Phylum','Order','Class','Family','Genus'])
    eval_df_3 =  pd.DataFrame(np.zeros((5,len(lengths))),columns=lengths,index=['Phylum','Order','Class','Family','Genus'])

    y_sim_1,y_sim_2,y_sim_3 = data['phylum'].values,data['class_'].values,data['order'].values
    y_sim_4,y_sim_5 = data['family'].values,data['genus'].values
    for len_ in range(len(lengths[:-1])):
        gen = simulate_ngs_generator_fixed_len(data,lengths[len_],batch_size,max_len)
        y_1,y_2,y_3,y_4,y_5 = model.predict_generator(gen,verbose=1,steps=data.shape[0]//batch_size)

        for i in range(5):
            y_pred = [y_1,y_2,y_3,y_4,y_5][i].argmax(axis = -1).astype('uint16')
            for metric in metrics:
                if metric == 'f1':
                    eval_df_1.iloc[i,len_] = f1_score([y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5][i], y_pred,average='micro')
                    eval_df_1.to_csv(save_path+'_'+str(metric)+'.csv')  
                elif metric == 'precision':
                    eval_df_2.iloc[i,len_] = precision_score([y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5][i], y_pred,average='micro')
                    eval_df_2.to_csv(save_path+'_'+str(metric)+'.csv')  
                
                else:
                    eval_df_3.iloc[i,len_] = recall_score([y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5][i], y_pred,average='micro')
                    eval_df_3.to_csv(save_path+'_'+str(metric)+'.csv')  

    #prediting on the full length
    data['simulated'] = pad_sequences(data['encoded'].values,maxlen=max_len).tolist()
    x_sim = data['simulated']
    x_sim = array(np.concatenate(x_sim.values).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
    y_1,y_2,y_3,y_4,y_5 = model.predict(x_sim,verbose=1,batch_size = batch_size)
    for i in range(5):
        y_pred = [y_1,y_2,y_3,y_4,y_5][i].argmax(axis = -1).astype('uint16')
        for metric in metrics:
            if metric == 'f1':
                eval_df_1.iloc[i,-1] = f1_score([y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5][i], y_pred,average='micro')
                eval_df_1.to_csv(save_path+'_'+str(metric)+'.csv')  
            elif metric == 'precision':
                eval_df_2.iloc[i,-1] = precision_score([y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5][i], y_pred,average='micro')
                eval_df_2.to_csv(save_path+'_'+str(metric)+'.csv')  
            else:
                eval_df_3.iloc[i,-1] = recall_score([y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5][i], y_pred,average='micro')
                eval_df_3.to_csv(save_path+'_'+str(metric)+'.csv')  
    return eval_df_1,eval_df_2,eval_df_3
            
def evaluate_ResNet_models(input_types,test,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    for x in range(len(input_types)):

        if input_types[x] == 'ResNet_SK':

            #ResNet_DC_No_W2V
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_DC_No_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_DC_No_W2V.csv'),max_len=max_len)
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
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_resnet(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_DC_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_DC_W2V.csv'),max_len=max_len)
        else:

            #ResNet SK
            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_SK.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_SK.csv'),max_len=max_len)



def evaluate_MLP_models(input_types,test,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    for x in range(len(input_types)):

        if input_types[x] == 'MLP_SK':
            #MLP_DC_No_W2V
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_DC_No_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_DC_No_W2V.csv'),max_len=max_len)

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
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_mlp(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_DC_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_DC_W2V.csv'),max_len=max_len)
        else:
            #MLP SK
            model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_SK.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_SK.csv'),max_len=max_len)

def evaluate_GRU_models(input_types,test,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    for x in range(len(input_types)):

        if input_types[x] == 'GRU_SK':
            #GRU_DC_No_W2V
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_DC_No_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_DC_No_W2V.csv'),max_len=max_len)


        elif input_types[x] == 'GRU_W2V':

            #GRU W2V
            model = build_gru(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_W2V.csv'),max_len=max_len)
        elif input_types[x] == 'GRU_SK_fixed_len':
            #GRU_SK_fixed_len
            model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_SK_Fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_SK_Fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'GRU_W2V_fixed_len':
            #GRU_W2V_fixed_len
            model = build_gru(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_W2V_fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_W2V_fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'GRU_DC_W2V':
            #GRU_DC_W2V
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_gru(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_DC_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_DC_W2V.csv'),max_len=max_len)
        else:
            #GRU SK
            model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_SK.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_SK.csv'),max_len=max_len)



def evaluate_mlp_only(database,embedding_matrix,max_len,kmer_size,metrics,test,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ### test  = pd.read_pickle(database+'/test.pkl')
    ### test  = test.sample(frac=1).reset_index(drop=True)
    #test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    #MLP SK
    model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.load_weights(database+'/models/MLP_SK.hdfs')
    evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                             batch_size=batch_size,save_path=''.join(database+'/results/MLP_SK.csv'),
                                             max_len=max_len,classes_1=classes_1,classes_2=classes_2,classes_3=classes_3,
                                             classes_4=classes_4,classes_5=classes_5,classes_6=classes_6)

def evaluate_gru_only(database,embedding_matrix,max_len,kmer_size,metrics,test,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    #MLP SK
    model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.load_weights(database+'/models/GRU_SK.hdfs')
    evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                             batch_size=batch_size,save_path=''.join(database+'/results/GRU_SK.csv'),
                                             max_len=max_len,classes_1=classes_1,classes_2=classes_2,classes_3=classes_3,
                                             classes_4=classes_4,classes_5=classes_5,classes_6=classes_6)

def evaluate_resnet_only(database,embedding_matrix,max_len,kmer_size,metrics,test,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    #test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    #MLP SK
    model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.load_weights(database+'/models/ResNet_SK.hdfs')
    evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,
                                             batch_size=batch_size,save_path=''.join(database+'/results/ResNet_SK.csv'),
                                             max_len=max_len,classes_1=classes_1,classes_2=classes_2,classes_3=classes_3,
                                             classes_4=classes_4,classes_5=classes_5,classes_6=classes_6)
#Evaluation function for the multi-task learning
def evaluate_multitask_on_different_length(model, lengths,data,batch_size,save_path,max_len,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    eval_df =  pd.DataFrame(np.zeros((5,len(lengths))),columns=lengths,index=['Phylum','Order','Class','Family','Genus'])
    y_sim_1,y_sim_2,y_sim_3 = data['phylum'].values,data['class_'].values,data['order'].values
    y_sim_4,y_sim_5 = data['family'].values,data['genus'].values
    for len_ in range(len(lengths[:-1])):
        gen = simulate_multi_task_generator_fixed_len(data,lengths[len_],batch_size,max_len,
            classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
        pred = model.predict_generator(gen,verbose=2,steps=data.shape[0]//batch_size)
        y_1,y_2,y_3,y_4,y_5 = np.hsplit(pred,np.array([classes_1,classes_2,classes_3,classes_4]))
        """     score = 0.0
        for class1 in Label1: #y_1
            for class2 in Label2: #y_2
                if class2 is ymshe with class1:
                    score = max(score, pred[class1] + pred[class2])"""
        for i in range(5):
            y_pred = [y_1,y_2,y_3,y_4,y_5][i].argmax(axis = -1).astype('uint16')
            eval_df.iloc[i,len_] = f1_score([y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5][i], y_pred,average='micro')
    eval_df.to_csv(save_path)  

    #prediting on the full length
    data['simulated'] = pad_sequences(data['encoded'].values,maxlen=max_len).tolist()
    x_sim = data['simulated']
    x_sim = array(np.concatenate(x_sim.values).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
    pred = model.predict(x_sim,verbose=2,batch_size = batch_size)
    y_1,y_2,y_3,y_4,y_5 = np.hsplit(pred,np.array([classes_1,classes_2,classes_3,classes_4]))
    for i in range(5):
        y_pred = [y_1,y_2,y_3,y_4,y_5][i].argmax(axis = -1)
        eval_df.iloc[i,-1] = precision_score([y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5][i], y_pred, average='micro')
    eval_df.to_csv(save_path)    
    return eval_df

#Evaluation is based on AUC
def evaluate_multi_task(database,embedding_matrix,max_len,kmer_size,metrics,test,batch_size,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    #MLP SK
    model = build_multitask_mlp(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.load_weights(database+'/models/Multitask_MLP_SK.hdfs')
    evaluate_multitask_on_different_length(model=model,lengths=[25,50,75,100,125,'full_HVR'],data=test,max_len=max_len,
                                             batch_size=batch_size,save_path=''.join(database+'/results/MLP_SK.csv'),
                                             classes_1=classes_1,classes_2 =classes_2,classes_3=classes_3,
                                             classes_4=    classes_4,classes_5 = classes_5,classes_6=classes_6)
