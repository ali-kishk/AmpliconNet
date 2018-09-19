from __future__ import print_function
from lib import *
import pandas as pd
import dask.dataframe as dd
from dask.multiprocessing import get
from Bio import Seq, SeqIO
from Bio.Alphabet import generic_dna
import numpy as np
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
from numpy import array


### Function for Nucleotide sequence one hot encoding
#1 Declare the alphabet
alphabet = 'ACGTNRYSWKMBDHV'
integer = [1,2,3,4,0]
#2 Declare mapping functions
char_to_int = {'A':1,'C':2,'G':3,'T':4,'R':5,'Y':6,'S':7,
               'W':8,'K':9,'M':10,'B':11,'D':12,'H':13,'V':14,'N':15}
int_to_char = {1:'A',2:'C',3:'G',4:'T',5:'R',6:'Y',7:'S',
               8:'W',9:'K',10:'M',11:'B',12:'D',13:'H',14:'V',15:'N'}


def map_silva_ids(HVR_path,SILVA_header):
	### Reading the HVR fasta file
	#Reading the database as a panda dataframe
	reads=[]
	for record in SeqIO.parse(HVR_path, "fasta"):
	    id_ = str(record.description).split(" ")[2].split(":")[0]
	    encoded = record.seq 
	    reads.append([id_,encoded,len(record.seq)])    
	    
	df = pd.DataFrame(reads,columns=['id','seq','len'])

	#Reading SILVA header to map their ranks by ID

	SILVA_header = pd.read_csv(SILVA_header,index_col=0)
	SILVA_header.index = SILVA_header['id']


	df['phylum']  = df['id'].map(SILVA_header['phylum'])
	df['class_']  = df['id'].map(SILVA_header['class_'])
	df['order']   = df['id'].map(SILVA_header['order'])
	df['family']  = df['id'].map(SILVA_header['family'])
	df['genus']   = df['id'].map(SILVA_header['genus'])
	df['species'] = df['id'].map(SILVA_header['species'])
	return df

def main():
	script = sys.argv[0]
	HVR_multifasta_path = sys.argv[1]
	SILVA_header = sys.argv[2]
	HVR_dir = sys.argv[3]
	#valid_path = sys.argv()[4]
	os.mkdir(HVR_dir)
	print('\n Step 1: Mapping the taxonomic ranks from the header dataframe to your database \n')

	df = map_silva_ids(HVR_multifasta_path,SILVA_header)

	df = encode_label_to_close_int(df,''.join(HVR_dir+'/genus_mapping.pkl'),
	                               ''.join(HVR_dir+'/species_mapping.pkl'),encode_species = True)

	df = df[df['phylum-']!=-1]
	df = df.drop(columns=[ 'phylum', 'class_', 'order', 'family','genus', 'species'])

	print('\n Step 2: Removing ambiguities characters by random coressponding A,C,G,T \n')
	d = Seq.IUPAC.IUPACData.ambiguous_dna_values
	ambiguous_ch = d.keys()- ['A','G','C','T']
	df['ambiguity_count'] = df['seq'].apply(lambda x: sum([''.join(x).count(y) for y in ambiguous_ch]))

	#df.to_pickle('PCR_SILVA_all/data/V2_df_without_encoding.pkl')
	
	# ### Reading V2 dataframe after preprocessing
	#df = pd.read_pickle('PCR_SILVA_all/data/V2_df_without_encoding.pkl')
	df_amb = df[df['ambiguity_count']>0]
	df_amb = expend_ambiguity_df(df_amb)
	df = df[df['ambiguity_count']==0]
	df['seq-'] = df['seq']
	df = pd.concat([df,df_amb])
	df = df.drop(columns=['seq'])
	# Applying encoding
	df['encoded'] = df['seq-'].apply(encode_nu)
	df = df.drop(columns=['seq-'])
	df = df.drop(columns=[ 'len', 'Complete_genus'])
	#df.to_pickle('PCR_SILVA_all/data/V2_df_after_encoding_&_ambiguity_exp.pkl')

	# ### Apply stratified sampling to have at least equal classes in Train, Test, Validation data
	#df = pd.read_pickle('PCR_SILVA_all/data/V2_df_after_encoding_&_ambiguity_exp.pkl')

	df = df.sample(frac=1).reset_index(drop=True)

	counts = df['species-'].value_counts()
	df = df[df['species-'].isin(counts[counts > 2].index)]

	print('\n Step 3: Stratified sampling to Train, Tet, validation datasets\n')
	train,test,_,_ = train_test_split(df,df['species-'],random_state= 1,test_size = 0.2)
	train,valid,_,_ = train_test_split(train,train['species-'],random_state= 1,test_size = 0.15)
	train = train[train['species-'].isin(unique(valid['species-']))]
	test = test[test['species-'].isin(unique(valid['species-']))]
	train = train[train['species-'].isin(unique(test['species-']))]
	valid = valid[valid['species-'].isin(unique(test['species-']))]
	test  = test[test['species-'].isin(unique(train['species-']))]
	valid = valid[valid['species-'].isin(unique(train['species-']))]
	# Reducing labels to number of integers in all train, test , valid
	train['type'] = 'train'
	test['type'] = 'test'
	valid['type'] = 'valid'

	df = pd.concat([train,test,valid])
	df = df.sort_values(by=['phylum-'])
	df['phylum-']= pd.factorize(df['phylum-'])[0]
	df = df.sort_values(by=['class_-'])
	df['class_-']= pd.factorize(df['class_-'])[0]
	df = df.sort_values(by=['order-'])
	df['order-']= pd.factorize(df['order-'])[0]
	df = df.sort_values(by=['family-'])
	df['family-']= pd.factorize(df['family-'])[0]
	df = df.sort_values(by=['genus-'])
	df['genus-']= pd.factorize(df['genus-'])[0]
	species_mapping_2 = pd.factorize(df['Complete_species'])[1]

	with open(HVR_dir+'/species_mapping_2.pkl', 'wb') as fp:
	    pickle.dump(species_mapping_2, fp)

	df['species-']= pd.factorize(df['Complete_species'])[0]

	train = df[df['type']=='train'].drop(columns=['type'])
	test  =  df[df['type']=='test'].drop(columns=['type'])
	valid = df[df['type']=='valid'].drop(columns=['type'])

	valid.to_pickle(HVR_dir+'/valid.pkl')
	test.to_pickle(HVR_dir+'/test.pkl')
	train.to_pickle(HVR_dir+'/train.pkl')

main()