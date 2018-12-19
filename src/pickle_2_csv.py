import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='pickle_2_csv: Converting a dabase of pickle files to csv for training fom disk for\\\
 saving memory')

parser.add_argument('--database_dir', dest='database_dir',type=str, default='.', help='Input directory contains train, test & valid data')

# Parameters
args = parser.parse_args()
database = args.database_dir

pkl_files = os.listdir(database)
pkl_files = [x for x in pkl_files if x.endswith('.pkl')]
pkl_files = [x for x in pkl_files if not x.endswith('_mapping.pkl')]

for file in pkl_files:
    df = pd.read_pickle(database+'/'+file)
    name = str(database+'/'+file.split('.')[0]+'.csv')
    print(name)
    df.to_csv(name)