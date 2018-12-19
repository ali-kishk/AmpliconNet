
# coding: utf-8

# ### Expanding ambigous nucleotide characters with ACTG f',' easier insilico PCR with ipcress

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#HVR = pd.read_csv('../16s_HVR_universal_primers.csv')
#HVR.columns = ['ind','primer_id','forward_seq','primer_id_2','reverse_seq','region']
#HVR = HVR.drop(columns=['ind','primer_id','primer_id_2'])


# In[133]:


HVR = pd.read_csv('Insilico_PCR/HVR_ipcress.tsv',sep=' ')


# In[1]:


from Bio import Seq
from itertools import product

def extend_ambiguous_dna(seq):
    """return list of all possible sequences given an ambiguous DNA input"""
    d = Seq.IUPAC.IUPACData.ambiguous_dna_values
    return  list(map("".join, product(*map(d.get, seq)))) 


# In[3]:


extend_ambiguous_dna('NW')


# In[136]:


d = Seq.IUPAC.IUPACData.ambiguous_dna_values
ambiguous_ch = d.keys()- ['A','G','C','T']
ambiguous_ch


# In[138]:


extended_df =  pd.DataFrame(columns = ['id','primer_A','primer_B','min_product_len','mix_product_len'])
for i in range(HVR.shape[0]):
    primer = HVR.iloc[i,1].upper()
    if [primer.find(x) for x in ambiguous_ch] == [-1]*12:
        extended_df = pd.concat([extended_df,pd.DataFrame(HVR.iloc[i,:]).transpose()])
    else:
        ext_primers = extend_ambiguous_dna(primer)
        new_df = pd.DataFrame(np.zeros((len(ext_primers),5)))
        new_df.columns = ['id','primer_A','primer_B','min_product_len','mix_product_len']
        new_df['primer_A'] = ext_primers
        new_df['primer_B'] = HVR.iloc[i,2].upper()
        new_df['id'] = HVR.iloc[i,0]
        new_df['min_product_len'] = HVR.iloc[i,3]
        new_df['mix_product_len'] = HVR.iloc[i,4]
        extended_df = pd.concat([extended_df,new_df])


# In[139]:


HVR = extended_df


# In[140]:


extended_df =  pd.DataFrame(columns = ['id','primer_A','primer_B','min_product_len','mix_product_len'])
for i in range(HVR.shape[0]):
    primer = HVR.iloc[i,2].upper()
    if [primer.find(x) for x in ambiguous_ch] == [-1]*12:
        extended_df = pd.concat([extended_df,pd.DataFrame(HVR.iloc[i,:]).transpose()])
    else:
        ext_primers = extend_ambiguous_dna(primer)
        new_df = pd.DataFrame(np.zeros((len(ext_primers),5)))
        new_df.columns = ['id','primer_A','primer_B','min_product_len','mix_product_len']
        new_df['primer_B'] = ext_primers
        new_df['primer_A'] = HVR.iloc[i,1].upper()
        new_df['id'] = HVR.iloc[i,0]
        new_df['min_product_len'] = HVR.iloc[i,3]
        new_df['mix_product_len'] = HVR.iloc[i,4]
        extended_df = pd.concat([extended_df,new_df])


# In[143]:


extended_df.to_csv('Insilico_PCR/HVR_ipcress_extended.tsv',sep=' ',index=False)


# In[149]:


extended_df['min_product_len'] = 50
extended_df['mix_product_len'] = 1400


# In[150]:


extended_df['max_product_len'] = extended_df['mix_product_len']


# In[151]:


extended_df = extended_df.drop(columns=['mix_product_len'])


# In[153]:


extended_df.to_csv('Insilico_PCR/HVR_ipcress_extended_max_range.tsv',sep=' ',index=False)

