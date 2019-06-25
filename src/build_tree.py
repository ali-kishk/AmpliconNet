## Build a tree from any pickle dataset to make hierarchy base prediction using the prediction probability
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='AmpliconNet: Sequence Based Multi-layer Perceptron for Amplicon Read Classifier')
parser.add_argument('--database_dir', dest='database_dir',type=str, default='.', help='Input directory contains train, test & valid data')

args = parser.parse_args()
database = args.database_dir


df = pd.read_pickle(database+'/valid.pkl')
tree = df.iloc[:,1:-1].drop_duplicates()
tree.to_csv(database+'/tree.csv')
