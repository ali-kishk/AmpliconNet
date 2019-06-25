from __future__ import print_function
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
from lib import *
from train_func import *
import argparse

parser = argparse.ArgumentParser(description='AmpliconNet: Sequence Based Multi-layer Perceptron for Amplicon Read Classifier')
parser.add_argument('--tree_infer', dest='tree_infer', type=int, default=1, help='checking if the predicted genus follow the other output hierarchy.')
parser.add_argument('--pred_dir', dest='pred_dir',type=str, default='.', help='Directory path of the prediction files')
parser.add_argument('--o-taxa_table', dest='taxa_table_path',type=str, default='.', help='Path for the output taxonomy table')
parser.add_argument('--target_rank', dest='target_rank',type=str, default='genus', help='Target taxonomy rank to report the taxonomy table,eg: genus, family, class, order, phylum, all')
parser.add_argument('--min_prob', dest='min_prob', type=float, default=0.08, help='Minimum prediction probability as a cutoff')
parser.add_argument('--min_count', dest='min_count', type=int, default=6, help='Minimum number of reported read per class to report in the taxonomy table [0: False, 1:True].')
parser.add_argument('--biom_taxon_table', dest='biom_taxon_table', type=bool, default=False, help='Whetether to output biom taxonomy table to be converted to BIOM format ')

args = parser.parse_args()
pred_dir = args.pred_dir
MIN_COUNT = args.min_count
MIN_PROB = args.min_prob
TARGET_RANK = args.target_rank
taxa_table_path = args.taxa_table_path
tree_infer= int(args.tree_infer)
biom_taxon_table = args.biom_taxon_table


pred_files = os.listdir(pred_dir)
pred_files = [x for x in pred_files if x.endswith('_AmpliconNet_taxonomy.csv')]


def unexpand_genus(row,str_):
    return row['genus'].split(str_)[1].split('__')[0]

def check_genus_name(df):
    """A function to check if the genus name is "uncultered" to add it's order
    """
    if df['genus_'] == 'uncultured':
        genus = ''.join(df['family_']+'__'+df['genus_'])
    else:
        genus = df['genus_']
    return genus

def check_family_name(df):
    """A function to check if the family name is "Unknown Family" to add it's order
    """
    if df['family_'] == 'Unknown Family':
        family = ''.join(df['order_']+'__'+df['family_'])
    else:
        family = df['family_']
    return family

final_df = pd.DataFrame()
for file in pred_files:
    df = pd.read_csv(pred_dir+'/'+file)
    df = df[df['id']!='DUMPY_SEQ']
    df = df.drop(columns=['id','Unnamed: 0'])
    df = df[df['genus_prob']>=MIN_PROB]
    df = df[df['family_prob']>=MIN_PROB]
    df = df[df['order_prob']>=MIN_PROB]
    df = df[df['class_prob']>=MIN_PROB]
    df = df[df['phylum_prob']>=MIN_PROB]

    #checking if the predicted genus follow the other output
    df['phylum_'] = df.apply(lambda row:unexpand_genus(row,'P_'),axis=1)
    df['class_'] = df.apply(lambda row:unexpand_genus(row,'__C_'),axis=1)
    df['order_'] = df.apply(lambda row:unexpand_genus(row,'__O_'),axis=1)
    df['family_'] = df.apply(lambda row:unexpand_genus(row,'__F_'),axis=1)
    df['genus_'] = df.apply(lambda row:unexpand_genus(row,'__G_'),axis=1)
    if tree_infer == 1:
        df.loc[df['family_']!=df['family'],['genus_','family_']] = 'UNK'
        df.loc[df['order_']!=df['order'],['genus_','family_','order_']] ='UNK'
        df.loc[df['class_']!=df['class'],['genus_','family_','order_','class_']] ='UNK'
        df = df[df['phylum']==df['phylum_']]
    else:
        pass
    df['genus_'] = df.apply(lambda row: check_genus_name(row), axis=1)
    df['family_'] = df.apply(lambda row: check_family_name(row), axis=1)
    df = df.iloc[:,-5:]
    df['file'] = str(file.split('_AmpliconNet_taxonomy.csv')[0])
    #df.loc[df['phylum_']!=df['phylum'],['genus_','family_','order_','class_']] ='UNK'
    """df = df[df['family']==df['family_']]
    df = df[df['phylum']==df['phylum_']]
    df = df[df['class']==df['class_']]
    df = df[df['order']==df['order_']]"""

    # convert taxonomy table to biom format
    if biom_taxon_table:
        df['phylum_'] = "k__Bacteria;p__" + df['phylum_']
        df['class_'] = df['phylum_'] + ';c__' + df['class_']
        df['order_'] = df['class_'] + ';o__' + df['order_']
        df['family_'] = df['order_'] + ';f__' + df['family_']
        df['genus_'] = df['family_'] + ';g__' + df['genus_']

    final_df = pd.concat([final_df,df])

pred_files = [x.split('_AmpliconNet_taxonomy.csv')[0] for x in pred_files if x.endswith('_AmpliconNet_taxonomy.csv')]
pd.DataFrame(final_df.groupby(final_df.columns.tolist(),as_index=False).size()).to_csv('temp')

count_table = pd.read_csv('temp')
if TARGET_RANK !='all':

        TARGET_RANK = ''.join(TARGET_RANK+'_')
        final_df = final_df[final_df[TARGET_RANK]!='UNK']
        taxa = final_df[TARGET_RANK].unique()
        num_taxa = len(taxa)

        taxa_table = pd.DataFrame(data=np.zeros((len(pred_files),num_taxa)),index=pred_files,columns=taxa)
        for i in range(count_table.shape[0]):
            rank = count_table.loc[i,TARGET_RANK]
            file = count_table.loc[i,'file']
            taxa_table.loc[file,rank] = count_table.loc[i,'0']

            # convert taxonomy table to biom format
            if biom_taxon_table:
                taxa_table = taxa_table.transpose()
                taxa_table.index.name = 'Taxon_ID'
                taxa_table.to_csv(taxa_table_path,sep='\t')
            else:
                taxa_table.to_csv(taxa_table_path)
else:
        final_df_ = final_df
        for x in ['phylum_','class_','order_','family_','genus_']:
                TARGET_RANK = x
                print(x)
                final_df_ = final_df_[final_df_[TARGET_RANK]!='UNK']
                taxa = final_df[TARGET_RANK].unique()
                num_taxa = len(taxa)
                taxa_table = pd.DataFrame(data=np.zeros((len(pred_files),num_taxa)),index=pred_files,columns=taxa)
                final_taxa_table = pd.DataFrame(index=pred_files)
                for i in range(count_table.shape[0]):
                    rank = count_table.loc[i,TARGET_RANK]
                    file = count_table.loc[i,'file']
                    taxa_table.loc[file,rank] = count_table.loc[i,'0']
                final_taxa_table = pd.concat([final_taxa_table,taxa_table],axis=1)

                # convert taxonomy table to biom format
                if biom_taxon_table:
                    final_taxa_table = final_taxa_table.transpose()
                    final_taxa_table.index.name = 'Taxon_ID'
                    final_taxa_table.to_csv(taxa_table_path,sep='\t')
                else:
                    final_taxa_table.to_csv(taxa_table_path)