from __future__ import print_function
from lib import *
import pandas as pd
import dask.dataframe as dd
from dask.multiprocessing import get
from Bio import Seq, SeqIO
from Bio.Alphabet import generic_dna
import numpy as np
from numpy import array
import pandas as pd
from random import randint, random,sample
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math
from numpy import unique
import pickle
from sklearn.metrics import r2_score, accuracy_score
from itertools import product
import os
import sys
from random import randint, random,sample


## Functions to concatenate the labels from phylum to genus as some sequences has repeated genus over different families
def from_phylum_to_genus(df):
    class__ = ''.join(#'K_'+str(df['kingdom'])+'__'+
                      'P_'+str(df['phylum'])+'__'+
                      'C_'+str(df['class_'])+'__'+
                      'O_'+str(df['order'])+'__'+
                      'F_'+str(df['family'])+'__'+
                      'G_'+str(df['genus']))
    return class__


def from_phylum_to_species(df):
    class__ = ''.join(#'K_'+str(df['kingdom'])+'__'+
                      'P_'+str(df['phylum'])+'__'+
                      'C_'+str(df['class_'])+'__'+
                      'O_'+str(df['order'])+'__'+
                      'F_'+str(df['family'])+'__'+
                      'G_'+str(df['genus'])+'__'+
                      'S_'+str(df['species']))
    return class__


def encode_label_to_close_int(df,genus_path,species_path,encode_species = True):   
    """
    input:
    df: dataframe of SILVA_taxa that
    genus_path :  path to save the pkl file of teh genus mapping
    species_path :  path to save the pkl file of teh species mapping

    
    output:
    df: the same dataframe with additional columns that represent one hot encoding 
                  that takes into account the hierarchical structure
    
    """
    """
    kingdoms_ = df['kingdom'].value_counts().index
    dict_ = dict()
    dict_ = dict_.fromkeys(kingdoms_)
    keys = range(len(kingdoms_))
    for k in keys:
        dict_[kingdoms_[k]] = keys[k]
    #df['kingdom-'] = df['kingdom'].apply(lambda x: dict_[x])
    df['Domain-'] = pd.factorize(df['Domain'].values)[0]
    df = df.sort_values(by='Domain-')"""
    df['phylum-'] = pd.factorize(df['phylum'].values)[0]
    df = df.sort_values(by=['phylum-'])
    df['class_-'] = pd.factorize(df['class_'].values)[0]
    df.sort_values(by=['class_-'])
    df['order-'] = pd.factorize(df['order'].values)[0]
    df.sort_values(by=['order-'])
    df['family-'] = pd.factorize(df['family'].values)[0]
    df.sort_values(by=['family-'])    
    #df = df[df['genus']!='uncultured']
    ## As some genus isn "unidetified" across many families, so 
    ## concatenate SILVA labels to one label from kingdom to genus
    ddata = dd.from_pandas(df, npartitions=30)
    df['Complete_genus'] = ddata.map_partitions(lambda df:df.apply(lambda row: 
                                                                   from_phylum_to_genus(row),
                                                                    axis=1)).compute(scheduler='threads')
    genus_mapping = pd.factorize(df['Complete_genus'].values)[1]
    df['genus-'] = pd.factorize(df['Complete_genus'].values)[0]
    df = df.sort_values(by=['genus-'])
    #save this mapping as a pickle file for all HVR models
    with open(genus_path, 'wb') as fp:
        pickle.dump(genus_mapping, fp)
    
    if encode_species == True:
        ddata = dd.from_pandas(df, npartitions=30)
        df['Complete_species'] = ddata.map_partitions(lambda df:df.apply(lambda row: 
                                                                                 from_phylum_to_species(row),
                                                                                 axis=1)).compute(scheduler='threads')
        species_mapping = pd.factorize(df['Complete_species'].values)[1]

        df['species-'] = pd.factorize(df['Complete_species'].values)[0]
        df = df.sort_values(by=['species-'])
        #save this mapping as a pickle file for all HVR models
        with open(species_path, 'wb') as fp:
            pickle.dump(species_mapping, fp)
    return df

d = Seq.IUPAC.IUPACData.ambiguous_dna_values
ambiguous_ch = d.keys()- ['A','G','C','T']

# Replacing any ambiguity characters with random corresponding Nu character
def expend_ambiguity_df(df_amb):
    
    df_amb = df_amb[df_amb['ambiguity_count']<10]
    df_amb = df_amb.reset_index().drop(columns=['index'])
    new_df = pd.DataFrame(columns=df_amb.columns)
    new_df['seq-'] = ''
    for i in range(df_amb.shape[0]):
        sample =  df_amb.iloc[i,:]
        num_moves = len(list(map("".join, product(*map(d.get, sample['seq'])))))
        moves = list(map("".join, product(*map(d.get, sample['seq']))))
        seq = sample['seq']
        if num_moves in [1,2,3]:
            new_sample = pd.DataFrame(columns=df_amb.columns)
            new_sample = new_sample.append([sample]*1)
            choice = randint(0,num_moves-1)
            
            x = moves[choice]#[x[y] for y in choice]
            new_sample['seq-'] = np.array(set(x))
            new_df = pd.concat([new_df,new_sample])
        else:
            new_sample = pd.DataFrame(columns=df_amb.columns)
            new_sample = new_sample.append([sample]*1)
            choice = randint(0,num_moves-1)
            #x = list(map("".join, product(*map(d.get, seq))))
            x = moves[choice]#[x[y] for y in choice]          
            new_sample['seq-'] = np.array(set(x))
            new_df = pd.concat([new_df,new_sample])
    return new_df

### Function for Nucleotide sequence one hot encoding
#1 Declare the alphabet
alphabet = 'ACGTNRYSWKMBDHV'
integer = [1,2,3,4,0]
#2 Declare mapping functions
char_to_int = {'A':1,'C':2,'G':3,'T':4,'R':5,'Y':6,'S':7,
               'W':8,'K':9,'M':10,'B':11,'D':12,'H':13,'V':14,'N':15}
int_to_char = {1:'A',2:'C',3:'G',4:'T',5:'R',6:'Y',7:'S',
               8:'W',9:'K',10:'M',11:'B',12:'D',13:'H',14:'V',15:'N'}
#3 convert char to number
def encode_nu(sequence):
    return array([char_to_int[char] for char in sequence])
