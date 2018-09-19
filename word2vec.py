 ### Training Word2Vec on encoded sequences
import sys
from gensim.models.word2vec import Word2Vec,LineSentence
from gensim.models import KeyedVectors
import multiprocessing
import pandas as pd
import numpy as np
import itertools
from gensim.models.word2vec import Word2Vec,LineSentence
from gensim.models import KeyedVectors
import multiprocessing
from train_func import oneHotEncoding_to_kmers

database = sys.argv[1]
kmer_size = int(sys.argv[2])
MAX_LEN = int(sys.argv[3])

bases=['1','2','3','4']
all_kmers = [''.join(p) for p in itertools.product(bases, repeat=kmer_size)]
word_to_int = dict()
word_to_int = word_to_int.fromkeys(all_kmers)
keys = range(1,len(all_kmers)+1)
for k in keys:
    word_to_int[all_kmers[k-1]] = keys[k-1]


def save_kmerized_corpus(path, kmer_size,train):
    with open(path, 'w') as file:
        for i in range(train.shape[0]):
            kmers = oneHotEncoding_to_kmers(train['encoded'].iloc[i],kmer_size)
            file.write(str(kmers).replace(',','').replace('[','').replace(']',''))
            file.write('\n')


def main():
	script = sys.argv[0]

	train = pd.read_pickle(''.join(database+'/train.pkl'))
	valid = pd.read_pickle(''.join(database+'/valid.pkl'))

	np.random.seed(1)
	train = train.reset_index(drop=True)
	valid = valid.reset_index(drop=True)

	df = pd.concat([train,valid])

	#Saving the corpus to the disk to save Memory
	save_kmerized_corpus(''.join(database+'/W2V_Kmerized_corpus.txt'),kmer_size,df)

	corpus = LineSentence(source=''.join(database+'/W2V_Kmerized_corpus.txt'))

	# Gensim Word2Vec model
	vector_size = 150
	window_size = 50
	max_features = 4**kmer_size + 1
	max_len = MAX_LEN

	# Create Word2Vec
	word2vec = Word2Vec(sentences=corpus,
	                    sg=1,
	                    max_vocab_size=max_features,
	                    sample=1e-5,
	                    size=vector_size, 
	                    window=window_size,
	                    hs=1,
	                    negative=20,
	                    iter=5,
	                    workers=multiprocessing.cpu_count(),
	                    seed=5)

	#word2vec.train(sentences=corpus,epochs=4,total_examples=word2vec.corpus_count)
	word2vec.wv.save_word2vec_format(''.join(database+'/W2V_model_'+str(kmer_size)+'_kmer.w2v'))

main()