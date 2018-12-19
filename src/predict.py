from __future__ import print_function
from lib import *
from train_func import *
import pandas as pd
import dask.dataframe as dd
from dask.multiprocessing import get
from Bio import Seq, SeqIO
from Bio.Alphabet import generic_dna
import numpy as np
import pandas as pd
from random import randint, random,sample
import pandas as pd
import math
from numpy import unique, array
import pickle
from itertools import product
import os
import sys
from random import randint, random,sample
from keras.preprocessing.sequence import pad_sequences

import argparse

parser = argparse.ArgumentParser(description='AmpliconNet: Sequence Based Multi-layer Perceptron for Amplicon Read Classifier')
parser.add_argument('--database', dest='database',type=str, default='.', help='High Variable Region database for the input data. Example: V2')
parser.add_argument('--input_type', dest='input_type',type=str, default='fastq', help='Input types if Fastq or Fasta')
parser.add_argument('--dir_path', dest='dir_path',type=str, default='.', help='Directory contains input Fastq files')
parser.add_argument('--model_param_path', dest='model_param_path',type=str, default='./models_output_dim.csv', help='path of the model parameters file generated by generate_model_dim_table.py')
parser.add_argument('--output_dir', dest='output_dir',type=str, default='.', help='Directory for the out taxonomy tables')
parser.add_argument('--model_path', dest='model_path',type=str, default='MLP_SK.hdfs', help='Model path for the HVR databse')

# Parameters
args = parser.parse_args()
dir_path = args.dir_path
database = args.database
model_param_path = args.model_param_path
output_dir = args.output_dir
model_path = args.model_path
input_type = args.input_type
#model_path = ''.join(database+'/models/'+model_path)

def read_fastq(file_path):
    reads=[]
    for record in SeqIO.parse(file_path, input_type):
        id_ = record.description
        seq = record.seq 
        reads.append([id_,seq])
    df = pd.DataFrame(reads,columns=['id','seq'])
    return df

def map_prob(df,y_5,column):
    return y_5[df['index'],df[column]]

def batch_predict(dir_path,model_param_path,model_path,database,output_dir):
    files_list =  np.sort(os.listdir(path=dir_path)).tolist()
    embedding_matrix = 0
    kmer_size = 6
    dim_table = pd.read_csv(model_param_path)
    dim_table = dim_table[dim_table['database'] == database]
    max_len   = int(dim_table['max_len'] )
    classes_1 = int(dim_table['phylum']  )
    classes_2 = int(dim_table['class_']  )
    classes_3 = int(dim_table['order']   )
    classes_4 = int(dim_table['family']  )
    classes_5 = int(dim_table['genus']   )
    classes_6 = int(dim_table['species'] )
    phylum_map = pd.read_pickle(database+'/phylum_mapping.pkl')
    class_map = pd.read_pickle(database+'/class_mapping.pkl')
    order_map = pd.read_pickle(database+'/order_mapping.pkl')
    family_map = pd.read_pickle(database+'/family_mapping.pkl')
    genus_map = pd.read_pickle(database+'/genus_mapping.pkl')
    #species_map = pd.read_pickle(database+'/species_mapping.pkl')
    if model_path.split('/')[-1] == 'MLP_SK.hdfs':
        model = build_mlp(True,embedding_matrix,max_len,kmer_size,'accuracy',classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    elif model_path.split('/')[-1] == 'MLP_GRU.hdfs':
        model = build_gru(True,embedding_matrix,max_len,kmer_size,'accuracy',classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    else:
        model = build_resnet(True,embedding_matrix,max_len,kmer_size,'accuracy',classes_1,classes_2,classes_3,classes_4,classes_5,classes_6)
    
    model.load_weights(model_path)
    word_to_int = get_word_to_int(kmer_size)

    for file in files_list:
        df = read_fastq(dir_path+'/'+file)
        #Removing ambiguity characters
        d = Seq.IUPAC.IUPACData.ambiguous_dna_values
        ambiguous_ch = d.keys()- ['A','G','C','T']
        df['ambiguity_count'] = df['seq'].apply(lambda x: sum([''.join(x).count(y) for y in ambiguous_ch]))
        df_amb = df[df['ambiguity_count']>0]
        df_amb = expend_ambiguity_df(df_amb)
        df = df[df['ambiguity_count']==0]
        df['seq-'] = df['seq']
        df = pd.concat([df,df_amb])
        df = df.drop(columns=['seq'])
        # Applying encoding
        df['encoded'] = df['seq-'].apply(encode_nu)
        df['encoded'] = df['encoded'].apply(lambda  x: oneHotEncoding_to_kmers(encoded_list=x,kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()
        #df['len'] = df['encoded'].apply(lambda x: len(x))
        df['encoded'] = df['encoded'].apply(lambda x: x[0:max_len])
        #Padding sequences
        X = pad_sequences(df['encoded'].values,maxlen= max_len)
        #print(X)
        X = array(np.concatenate(X).reshape(X.shape[0],max_len).tolist()).astype('uint16')
        # Begin prediction
        y_1,y_2,y_3,y_4,y_5 = model.predict(X,batch_size=250)
        df['phylum'],df['class'],df['order'],df['family'],df['genus'] = y_1.argmax(axis=-1) , y_2.argmax(axis=-1) , y_3.argmax(axis=-1) , y_4.argmax(axis=-1) , y_5.argmax(axis=-1)
        #Mapping the prediction probability for the genus level
        df['index'] = df.index
        df['phylum_prob'] = df.apply(lambda row:map_prob(row,y_1,'phylum'),axis=1)
        #df = df.sort_values(by=['phylum_prob'],ascending=False)
        #Mapping the prediction probability for the genus level
        df['class_prob'] = df.apply(lambda row:map_prob(row,y_2,'class'),axis=1)
        #df = df.sort_values(by=['class_prob'],ascending=False)
        #Mapping the prediction probability for the genus level
        df['order_prob'] = df.apply(lambda row:map_prob(row,y_3,'order'),axis=1)
        #df = df.sort_values(by=['order_prob'],ascending=False)
        #Mapping the prediction probability for the genus level
        df['family_prob'] = df.apply(lambda row:map_prob(row,y_4,'family'),axis=1)
        #df = df.sort_values(by=['family_prob'],ascending=False)
        #Mapping the prediction probability for the genus level
        df['genus_prob'] = df.apply(lambda row:map_prob(row,y_5,'genus'),axis=1)
        df = df.sort_values(by=['genus_prob'],ascending=False)
        df = df.drop(columns=['index'])
        #Mapping the classes names from the pkl files
        df['phylum'] = df['phylum'].apply(lambda x : phylum_map[x])
        df['class'] = df['class'].apply(lambda x : class_map[x])
        df['order'] = df['order'].apply(lambda x : order_map[x])
        df['family'] = df['family'].apply(lambda x : family_map[x])
        df['genus'] = df['genus'].apply(lambda x : genus_map[x])
        #df['species'] = df['species'].apply(lambda x : species_map[x])
        
        df = df.drop(columns=['ambiguity_count','seq-','encoded'])
        df.to_csv(output_dir+'/'+file+'_AmpliconNet_taxonomy.csv')


batch_predict(dir_path,model_param_path,model_path,database,output_dir)
