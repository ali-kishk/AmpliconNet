
# coding: utf-8

# In[155]:


import os
import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score,jaccard_similarity_score
from sklearn.metrics import f1_score,precision_score,recall_score

# In[2]:


# Parameters
parser = argparse.ArgumentParser(description='AmpliconNet: Sequence Based Multi-layer Perceptron for Amplicon Read Classifier')
parser.add_argument('--database_dir', dest='database_dir',type=str, default='.', help='Input directory contains train, test & valid data')
args = parser.parse_args()
database = args.database_dir


# In[144]:


pred_df = pd.read_csv(''.join('./'+database+'/test_pred_taxonomy.tsv'),sep='\t')


# In[145]:


pred_df['phylum'] = pred_df['Taxon'].apply(lambda x: x.split('; p__')[-1].split('; ')[0])
pred_df['class'] = pred_df['Taxon'].apply(lambda x: x.split('; c__')[-1].split('; ')[0])
pred_df['order'] = pred_df['Taxon'].apply(lambda x: x.split('; o__')[-1].split('; ')[0])
pred_df['family'] = pred_df['Taxon'].apply(lambda x: x.split('; f__')[-1].split('; ')[0])
pred_df['genus'] = pred_df['Taxon'].apply(lambda x: x.split('; g__')[-1].split('; ')[0])


# In[146]:


actual_df = pd.read_csv(''.join('./'+database+'/test_q2_taxonomy.txt'),sep='\t',header=None)


# In[147]:


actual_df.columns= ['id','Taxon']


# In[148]:


actual_df['phylum'] = actual_df['Taxon'].apply(lambda x: x.split('; p__')[-1].split('; ')[0])
actual_df['class'] = actual_df['Taxon'].apply(lambda x: x.split('; c__')[-1].split('; ')[0])
actual_df['order'] = actual_df['Taxon'].apply(lambda x: x.split('; o__')[-1].split('; ')[0])
actual_df['family'] = actual_df['Taxon'].apply(lambda x: x.split('; f__')[-1].split('; ')[0])
actual_df['genus'] = actual_df['Taxon'].apply(lambda x: x.split('; g__')[-1].split('; ')[0])


# In[122]:


#del_ind = pred_df[pred_df['genus']=='k__Bacteria'].index
#actual_df = actual_df.drop(axis=0,index=del_ind)
#pred_df = pred_df.drop(axis=0,index=del_ind)


# In[153]:


def evaluate_rdp(pred_df,actual_df,save_path):
    eval_df = pd.DataFrame(columns=['Phylum','Order','Class','Family','Genus'],
                           index=['F1_score','Precision','Recall'])
    metrices = ['F1_score']#,'Precision','Recall']
    for i in range(5):
        for metric in metrices:
            if metric == 'F1_score':
                eval_df.iloc[0,i] = f1_score(actual_df.iloc[:,2+i], pred_df.iloc[:,3+i],average='micro')
            elif metric == 'Precision':
                eval_df.iloc[1,i] = precision_score(actual_df.iloc[:,2+i], pred_df.iloc[:,3+i],average='micro')
            else:
                eval_df.iloc[2,i] = recall_score(actual_df.iloc[:,2+i], pred_df.iloc[:,3+i],average='micro')
    eval_df.to_csv(save_path+'_'+str(metric)+'.csv')  


# In[154]:


evaluate_rdp(pred_df,actual_df,save_path=''.join(database+'/results/RDP_summary'))

