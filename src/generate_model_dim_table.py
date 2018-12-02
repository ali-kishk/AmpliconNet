import numpy as np
import pandas as pd
import os
import glob

def generate_model_dim_table():
    files_list =  np.sort(glob.glob("*/train.pkl")).tolist()
    dim_table = pd.DataFrame(np.zeros((len(files_list),8)),columns=['database','max_len','phylum','class_','order','family','genus','species'])

    for i in range(len(files_list)):
        database = files_list[i].split('/')[0]
        dim_table.iloc[i,0] = database
        train = pd.read_pickle(database+'/train.pkl')
        train['len'] = train['encoded'].apply(lambda x: len(x))
        dim_table.iloc[i,1] = max(train['len']) +1
        dim_table.iloc[i,2]  = max(train['phylum'])  +1
        dim_table.iloc[i,3]  = max(train['class_'])  +1
        dim_table.iloc[i,4]   = max(train['order'])   +1
        dim_table.iloc[i,5]  = max(train['family'])  +1
        dim_table.iloc[i,6]   = max(train['genus'])   +1
        #dim_table.iloc[i,7] = max(train['species-']) +1
        dim_table.to_csv('models_output_dim.csv')

generate_model_dim_table()
