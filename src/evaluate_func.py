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
from keras.layers import LSTM, TimeDistributed, AveragePooling1D, Flatten,Activation,ZeroPadding1D,SeparableConv1D, GlobalAveragePooling1D
from keras.optimizers import Adam, rmsprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint, CSVLogger
from keras.layers import Conv1D, GlobalMaxPooling1D, ConvLSTM2D, Bidirectional,RepeatVector
from keras.regularizers import *
from keras import regularizers
from keras.layers import concatenate as concatLayer
from keras.utils import plot_model

import itertools
from itertools import product
from joblib import Parallel, delayed

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
#prob_tree = True
# Precision evaluation on multi-output
def evaluate_multioutput_on_different_length(model, lengths,data,batch_size,save_path,max_len,prob_tree,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):

    num_steps = data.shape[0]//batch_size
    data = data.iloc[:num_steps*batch_size,:]
    metrics = ['f1', 'precision','recall']

    eval_df_1 =  pd.DataFrame(np.zeros((5,len(lengths))),columns=lengths,index=['Phylum','Order','Class','Family','Genus'])
    eval_df_2 =  pd.DataFrame(np.zeros((5,len(lengths))),columns=lengths,index=['Phylum','Order','Class','Family','Genus'])
    eval_df_3 =  pd.DataFrame(np.zeros((5,len(lengths))),columns=lengths,index=['Phylum','Order','Class','Family','Genus'])

    y_sim_1,y_sim_2,y_sim_3 = data['phylum'].values,data['class_'].values,data['order'].values
    y_sim_4,y_sim_5 = data['family'].values,data['genus'].values
    y_true = [y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5]
    tree = pd.read_csv(save_path.split('/')[0]+'/tree.csv')
    if prob_tree ==True:
        save_path = save_path+'_prob_tree'
    else:
        pass
    for len_ in range(len(lengths[:-1])):
        gen = simulate_ngs_generator_fixed_len(data,lengths[len_],batch_size,max_len)
        y_1,y_2,y_3,y_4,y_5 = model.predict_generator(gen,verbose=1,steps=data.shape[0]//batch_size)

        # Apply prediction probability binning
        if prob_tree ==True:
            y_pred = Parallel(n_jobs=-1)(delayed(pred_probability_to_tree_pred)
                (y_1[i],y_2[i],y_3[i],y_4[i],y_5[i],tree) for i in range(len(y_1)))
            y_pred = np.array(y_pred)
        # Simple Softmax binning
        else:
            y_1,y_2,y_3,y_4,y_5 = y_1.argmax(axis=-1) , y_2.argmax(axis=-1) , y_3.argmax(axis=-1) , y_4.argmax(axis=-1) , y_5.argmax(axis=-1)
            y_pred = np.vstack([y_1,y_2,y_3,y_4,y_5]).transpose()
        for i in range(5):
            for metric in metrics:
                if metric == 'f1':
                    eval_df_1.iloc[i,len_] = f1_score(y_true[i], y_pred[:,i],average='micro')
                    eval_df_1.to_csv(save_path+'_'+str(metric)+'.csv')  
                elif metric == 'precision':
                    eval_df_2.iloc[i,len_] = precision_score(y_true[i], y_pred[:,i],average='micro')
                    eval_df_2.to_csv(save_path+'_'+str(metric)+'.csv')  
                
                else:
                    eval_df_3.iloc[i,len_] = recall_score(y_true[i], y_pred[:,i],average='micro')
                    eval_df_3.to_csv(save_path+'_'+str(metric)+'.csv')  

    #prediting on the full length
    data['simulated'] = pad_sequences(data['encoded'].values,maxlen=max_len).tolist()
    x_sim = data['simulated']
    x_sim = array(np.concatenate(x_sim.values).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
    y_1,y_2,y_3,y_4,y_5 = model.predict(x_sim,verbose=1,batch_size = batch_size)

    if prob_tree ==True:
        y_pred = Parallel(n_jobs=-1)(delayed(pred_probability_to_tree_pred)
            (y_1[i],y_2[i],y_3[i],y_4[i],y_5[i],tree) for i in range(len(y_1)))
        y_pred = np.array(y_pred)
    else:
        y_1,y_2,y_3,y_4,y_5 = y_1.argmax(axis=-1) , y_2.argmax(axis=-1) , y_3.argmax(axis=-1) , y_4.argmax(axis=-1) , y_5.argmax(axis=-1)
        y_pred = np.vstack([y_1,y_2,y_3,y_4,y_5]).transpose()

    for i in range(5):
        for metric in metrics:
            if metric == 'f1':
                eval_df_1.iloc[i,-1] = f1_score(y_true[i], y_pred[:,i],average='micro')
                eval_df_1.to_csv(save_path+'_'+str(metric)+'.csv')  
            elif metric == 'precision':
                eval_df_2.iloc[i,-1] = precision_score(y_true[i], y_pred[:,i],average='micro')
                eval_df_2.to_csv(save_path+'_'+str(metric)+'.csv')  
            else:
                eval_df_3.iloc[i,-1] = recall_score(y_true[i], y_pred[:,i],average='micro')
                eval_df_3.to_csv(save_path+'_'+str(metric)+'.csv')  
    return eval_df_1,eval_df_2,eval_df_3
            
def evaluate_ResNet_models(input_types,test,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,prob_tree,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()
    tree = pd.read_csv(database+'/tree.csv')

    for x in range(len(input_types)):

        if input_types[x] == 'ResNet_SK':

            #ResNet_DC_No_W2V
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_DC_No_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_DC_No_W2V.csv'),max_len=max_len)
        elif input_types[x] == 'ResNet_W2V':

            #ResNet W2V
            model = build_resnet(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_W2V.csv'),max_len=max_len)
        elif input_types[x] == 'ResNet_SK_fixed_len':
            #ResNet_SK_fixed_len
            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_SK_Fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_SK_Fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'ResNet_W2V_fixed_len':
            #ResNet_W2V_fixed_len
            model = build_resnet(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_W2V_fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_W2V_fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'ResNet_DC_W2V':
            #ResNet_DC_W2V
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_resnet(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_DC_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_DC_W2V.csv'),max_len=max_len)
        else:

            #ResNet SK
            model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/ResNet_SK.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/ResNet_SK.csv'),max_len=max_len)


def evaluate_SepConv_models(input_types,test,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,prob_tree,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    for x in range(len(input_types)):

        if input_types[x] == 'SepConv_SK':

            #SepConv_DC_No_W2V
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_sepconv(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/SepConv_DC_No_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/SepConv_DC_No_W2V.csv'),max_len=max_len)
        elif input_types[x] == 'SepConv_W2V':

            #SepConv W2V
            model = build_sepconv(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/SepConv_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/SepConv_W2V.csv'),max_len=max_len)
        elif input_types[x] == 'SepConv_SK_fixed_len':
            #SepConv_SK_fixed_len
            model = build_sepconv(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/SepConv_SK_Fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/SepConv_SK_Fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'SepConv_W2V_fixed_len':
            #SepConv_W2V_fixed_len
            model = build_sepconv(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/SepConv_W2V_fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/SepConv_W2V_fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'SepConv_DC_W2V':
            #SepConv_DC_W2V
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_sepconv(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/SepConv_DC_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/SepConv_DC_W2V.csv'),max_len=max_len)
        else:

            #SepConv SK
            model = build_sepconv(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/SepConv_SK.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/SepConv_SK.csv'),max_len=max_len)


def evaluate_MLP_models(input_types,test,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,prob_tree,
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
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_DC_No_W2V.csv'),max_len=max_len)

        elif input_types[x] == 'MLP_W2V':

            #MLP W2V
            model = build_mlp(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_W2V.csv'),max_len=max_len)

        elif input_types[x] == 'MLP_SK_fixed_len':
            #MLP_SK_fixed_len
            model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_SK_fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_SK_fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'MLP_W2V_fixed_len':
            #MLP_W2V_fixed_len
            model = build_mlp(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_W2V_fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_W2V_fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'MLP_DC_W2V':
            #MLP_DC_W2V
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_mlp(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_DC_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_DC_W2V.csv'),max_len=max_len)
        else:
            #MLP SK
            model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/MLP_SK.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/MLP_SK.csv'),max_len=max_len)

def evaluate_GRU_models(input_types,test,database,embedding_matrix,max_len,kmer_size,metrics,batch_size,prob_tree,
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
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_DC_No_W2V.csv'),max_len=max_len)


        elif input_types[x] == 'GRU_W2V':

            #GRU W2V
            model = build_gru(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_W2V.csv'),max_len=max_len)
        elif input_types[x] == 'GRU_SK_fixed_len':
            #GRU_SK_fixed_len
            model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_SK_Fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_SK_Fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'GRU_W2V_fixed_len':
            #GRU_W2V_fixed_len
            model = build_gru(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_W2V_fixed_len.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_W2V_fixed_len.csv'),max_len=max_len)

        elif input_types[x] == 'GRU_DC_W2V':
            #GRU_DC_W2V
            ## test  = pd.read_pickle(database+'/test.pkl')
            ## test  = test.sample(frac=1).reset_index(drop=True)

            model = build_gru(False,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_DC_W2V.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_DC_W2V.csv'),max_len=max_len)
        else:
            #GRU SK
            model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
            model.load_weights(database+'/models/GRU_SK.hdfs')
            evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                                     batch_size=250,save_path=''.join(database+'/results/GRU_SK.csv'),max_len=max_len)



def evaluate_mlp_only(database,embedding_matrix,max_len,kmer_size,metrics,test,batch_size,prob_tree,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ### test  = pd.read_pickle(database+'/test.pkl')
    ### test  = test.sample(frac=1).reset_index(drop=True)
    #test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()
    tree = pd.read_csv(str(database)+'/tree.csv')
    #MLP SK
    model = build_mlp(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.load_weights(database+'/models/MLP_SK.hdfs')
    evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                             batch_size=batch_size,save_path=''.join(database+'/results/MLP_SK.csv'),
                                             max_len=max_len,classes_1=classes_1,classes_2=classes_2,classes_3=classes_3,
                                             classes_4=classes_4,classes_5=classes_5,classes_6=classes_6)

def evaluate_gru_only(database,embedding_matrix,max_len,kmer_size,metrics,test,batch_size,prob_tree,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    #MLP SK
    model = build_gru(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.load_weights(database+'/models/GRU_SK.hdfs')
    evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                             batch_size=batch_size,save_path=''.join(database+'/results/GRU_SK.csv'),
                                             max_len=max_len,classes_1=classes_1,classes_2=classes_2,classes_3=classes_3,
                                             classes_4=classes_4,classes_5=classes_5,classes_6=classes_6)

def evaluate_sepconv_only(database,embedding_matrix,max_len,kmer_size,metrics,test,batch_size,prob_tree,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    #test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    #MLP SK
    model = build_sepconv(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.load_weights(database+'/models/SepConv_SK.hdfs')
    evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                             batch_size=batch_size,save_path=''.join(database+'/results/SepConv_SK.csv'),
                                             max_len=max_len,classes_1=classes_1,classes_2=classes_2,classes_3=classes_3,
                                             classes_4=classes_4,classes_5=classes_5,classes_6=classes_6)


def evaluate_resnet_only(database,embedding_matrix,max_len,kmer_size,metrics,test,batch_size,prob_tree,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    #test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    #MLP SK
    model = build_resnet(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.load_weights(database+'/models/ResNet_SK.hdfs')
    evaluate_multioutput_on_different_length(model=model, lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,
                                             batch_size=batch_size,save_path=''.join(database+'/results/ResNet_SK.csv'),
                                             max_len=max_len,classes_1=classes_1,classes_2=classes_2,classes_3=classes_3,
                                             classes_4=classes_4,classes_5=classes_5,classes_6=classes_6)


#Evaluation function for the multi-task learning
def evaluate_multitask_on_different_length(model, lengths,data,batch_size,prob_tree,save_path,max_len,
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
def evaluate_multi_task(database,embedding_matrix,max_len,kmer_size,metrics,test,batch_size,prob_tree,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    ## test  = pd.read_pickle(database+'/test.pkl')
    ## test  = test.sample(frac=1).reset_index(drop=True)
    test['encoded'] = test['encoded'].apply(lambda x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()

    #MLP SK
    model = build_multitask_mlp(True,embedding_matrix,max_len,kmer_size,metrics,
        classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    model.load_weights(database+'/models/Multitask_MLP_SK.hdfs')
    evaluate_multitask_on_different_length(model=model,lengths=[25,50,75,100,125,'full_HVR'],data=test,prob_tree=prob_tree,max_len=max_len,
                                             batch_size=batch_size,save_path=''.join(database+'/results/MLP_SK.csv'),
                                             classes_1=classes_1,classes_2 =classes_2,classes_3=classes_3,
                                             classes_4=    classes_4,classes_5 = classes_5,classes_6=classes_6)

# A generator function of a fixed length subsequence that changes in each epoch
def simulate_ngs_generator_fixed_len(df,len1,batch_size,max_len):
    y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
    x_sim = df['encoded'].apply(lambda x : simulate_reads(x,len1=len1))
    x_sim = pad_sequences(x_sim.values,maxlen=max_len)
    x_sim = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
    samples_per_epoch = df.shape[0]
    number_of_batches = samples_per_epoch//batch_size
    counter=0

    while counter <=number_of_batches :
        X_batch = np.array(x_sim[batch_size*counter:batch_size*(counter+1)]).astype('uint16')
        y_1 = y_sim_1[batch_size*counter:batch_size*(counter+1)].astype('uint8')
        y_2 = y_sim_2[batch_size*counter:batch_size*(counter+1)].astype('uint8')
        y_3 = y_sim_3[batch_size*counter:batch_size*(counter+1)].astype('uint8')
        y_4 = y_sim_4[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        y_5 = y_sim_5[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        #y_6 = y_sim_6[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        counter += 1
        if counter ==number_of_batches:
            counter = 0
            del x_sim
            y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
            x_sim = df['encoded'].apply(lambda x : simulate_reads(x,len1=len1))
            x_sim = pad_sequences(x_sim.values,maxlen=max_len)
            x_sim = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
        yield X_batch,[y_1,y_2,y_3,y_4,y_5]

# convert a prediction probability to tree prediction
def pred_probability_to_tree_pred(y_1,y_2,y_3,y_4,y_5,tree):
    #Input : preiction probability of a sample
    # Output: Classes that foolow the hierarchy of the highest rank
    #y_1,y_2,y_3,y_4,y_5 =  y_pred[0],y_pred[1],y_pred[2],y_pred[3],y_pred[4] 
    y_1 = y_1.argmax(-1)
    y_2_ind = tree[tree['phylum']==y_1]['class_'].unique()
    y_2 = y_2[y_2_ind].argmax(-1)
    y_2 = y_2_ind[y_2]
    y_3_ind = tree[tree['class_']==y_2]['order'].unique()
    y_3 = y_3[y_3_ind].argmax(-1)
    y_3 = y_3_ind[y_3]    
    y_4_ind = tree[tree['order']==y_3]['family'].unique()
    y_4 = y_4[y_4_ind].argmax(-1)    
    y_4 = y_4_ind[y_4]
    y_5_ind = tree[tree['family']==y_4]['genus'].unique()
    y_5 = y_5[y_5_ind].argmax(-1)    
    y_5 = y_5_ind[y_5]
    return y_1,y_2,y_3,y_4,y_5