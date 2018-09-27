from __future__ import print_function
import numpy as np
import pandas as pd 
from numpy import array
from numpy.random import randint

import random

from sklearn.metrics import r2_score, accuracy_score

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Concatenate, LeakyReLU, concatenate,GRU, Bidirectional, MaxPool1D,GlobalMaxPool1D,add
from keras.layers import Dense, Embedding, Input, Masking, Dropout, MaxPooling1D,Lambda, BatchNormalization, Reshape
from keras.layers import LSTM, TimeDistributed, AveragePooling1D, Flatten,Activation,ZeroPadding1D
from keras.optimizers import Adam, rmsprop
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint, CSVLogger
from keras.layers import Conv1D, GlobalMaxPooling1D, ConvLSTM2D, Bidirectional,RepeatVector
from keras.regularizers import *
from keras import regularizers
from keras.layers import concatenate as concatLayer
from keras.utils import plot_model

import itertools
from itertools import product

from gensim.models import KeyedVectors

kmer_size = 6
vector_size = 128
max_features = 4**kmer_size + 1
bases=['1','2','3','4']
all_kmers = [''.join(p) for p in itertools.product(bases, repeat=kmer_size)]
word_to_int = dict()
word_to_int = word_to_int.fromkeys(all_kmers)
keys = range(1,len(all_kmers)+1)
for k in keys:
    word_to_int[all_kmers[k-1]] = keys[k-1]

# Creating embedding matrix from the word2vec model
def build_embedding_matrix(model_path,word_to_int):
    word2vec = KeyedVectors.load_word2vec_format(model_path)
    embedding_matrix = np.zeros((max_features,vector_size))
    for i in word_to_int.values():
        embedding_matrix[i,:] = word2vec.wv[str(i)]
    return embedding_matrix

# Convert the sequence of chracters to sequence of kmers
def oneHotEncoding_to_kmers(encoded_list,kmer_size):
    word_list = []
    ch_str = str(encoded_list.tolist()).replace(',','').replace('[','').replace(']','').replace(' ','')
    for i in range(len(ch_str) - kmer_size + 1):
        word_list.append(int(word_to_int[ch_str[ i : i + kmer_size ]]))
    return word_list

# Creating a simulated read of a fixed length
def simulate_reads(list1,len1):
    start = randint(len(list1)-len1)
    return list1[start:start+len1]

# Creating a simulated read of a minumum length = min_len & maximum length = max_len (maximum length in a given database) 
def simulate_reads_range(list1,min_len):
    random.seed(random.randint(1,1000))
    start = randint(len(list1)-min_len)
    end   = randint(start+min_len,len(list1))
    return list1[start:end]

# A generator function of a variable length subsequence that changes in each epoch
def simulate_ngs_generator(df,len1 ,batch_size,max_len):
    y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5,y_sim_6 = df['phylum-'].values,df['class_-'].values,df['order-'].values,df['family-'].values,df['genus-'].values,df['species-'].values
    x_sim = df['encoded'].apply(lambda x : simulate_reads_range(x,min_len=len1))
    x_sim = pad_sequences(x_sim.values,maxlen=max_len)
    x_sim = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
    samples_per_epoch = df.shape[0]
    number_of_batches = samples_per_epoch//batch_size
    counter=0

    while counter <=number_of_batches :
        X_batch = np.array(x_sim[batch_size*counter:batch_size*(counter+1)]).astype('uint16')
        y_1 = y_sim_1[batch_size*counter:batch_size*(counter+1)].astype('uint8')
        y_2 = y_sim_2[batch_size*counter:batch_size*(counter+1)].astype('uint8')
        y_3 = y_sim_3[batch_size*counter:batch_size*(counter+1)].astype('uint8')
        y_4 = y_sim_4[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        y_5 = y_sim_5[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        y_6 = y_sim_6[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        counter += 1
        if counter ==number_of_batches:
            counter = 0
            del x_sim,  y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5,y_sim_6
            y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5,y_sim_6 = df['phylum-'].values,df['class_-'].values,df['order-'].values,df['family-'].values,df['genus-'].values,df['species-'].values
            x_sim = df['encoded'].apply(lambda x : simulate_reads_range(x,min_len=len1))
            x_sim = pad_sequences(x_sim.values,maxlen=max_len)
            x_sim = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
        yield X_batch, [y_1,y_2,y_3,y_4,y_5,y_6]

# A generator function of a fixed length subsequence that changes in each epoch
def simulate_ngs_generator_fixed_len(df,len1,batch_size,max_len):
    y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5,y_sim_6 = df['phylum-'].values,df['class_-'].values,df['order-'].values,df['family-'].values,df['genus-'].values,df['species-'].values
    x_sim = df['encoded'].apply(lambda x : simulate_reads(x,len1=len1))
    x_sim = pad_sequences(x_sim.values,maxlen=max_len)
    x_sim = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
    samples_per_epoch = df.shape[0]
    number_of_batches = samples_per_epoch//batch_size
    counter=0

    while counter <=number_of_batches :
        X_batch = np.array(x_sim[batch_size*counter:batch_size*(counter+1)]).astype('uint16')
        y_1 = y_sim_1[batch_size*counter:batch_size*(counter+1)].astype('uint8')
        y_2 = y_sim_2[batch_size*counter:batch_size*(counter+1)].astype('uint8')
        y_3 = y_sim_3[batch_size*counter:batch_size*(counter+1)].astype('uint8')
        y_4 = y_sim_4[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        y_5 = y_sim_5[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        y_6 = y_sim_6[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        counter += 1
        if counter ==number_of_batches:
            counter = 0
            del x_sim
            y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5,y_sim_6 = df['phylum-'].values,df['class_-'].values,df['order-'].values,df['family-'].values,df['genus-'].values,df['species-'].values
            x_sim = df['encoded'].apply(lambda x : simulate_reads(x,len1=len1))
            x_sim = pad_sequences(x_sim.values,maxlen=max_len)
            x_sim = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
        yield X_batch,[y_1,y_2,y_3,y_4,y_5,y_6]

# AdamW optimizer function
from keras.optimizers import Optimizer
from keras import backend as K
import six
import copy
from six.moves import zip
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces

class AdamW(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Decoupled weight decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/index.html)
        - [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.wd = K.variable(weight_decay, name='weight_decay') # decoupled weight decay (2/4)
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p # decoupled weight decay (4/4)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# A function for building ResNet model
def build_resnet(Traniable_embedding,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):

	inp = Input(shape=(max_len,),dtype='uint16')
	max_features = 4**kmer_size +1
	if Traniable_embedding ==True:
		main = Embedding(5, 128)(inp)
	else:
		main = Embedding(max_features, vector_size, weights=[embedding_matrix],trainable=False)(inp)

	main = Conv1D(filters=64, kernel_size=3, padding='same')(main)
	i_l1 = MaxPooling1D(pool_size=2)(main)
	main = Conv1D(filters=64, kernel_size=3, padding='same')(main)
	main = BatchNormalization()(main)
	main = Activation('relu')(main)
	main = Conv1D(filters=64, kernel_size=3, padding='same')(main)

	main = concatenate([main, i_l1],axis=1)

	main = BatchNormalization()(main)
	main = Activation('relu')(main)
	main = MaxPooling1D(pool_size=2)(main)
	i_l1 = Conv1D(filters=128, kernel_size=1, padding='same')(main)

	main = Conv1D(filters=128, kernel_size=3, padding='same')(main)
	main = BatchNormalization()(main)
	main = Activation('relu')(main)
	main = Conv1D(filters=128, kernel_size=3, padding='same')(main)
	main = concatenate([main, i_l1],axis=1)
	main = BatchNormalization()(main)
	main = Activation('relu')(main)
	main = MaxPooling1D(pool_size=2)(main)
	i_l1 = Conv1D(filters=256, kernel_size=1, padding='same')(main)

	main = Conv1D(filters=256, kernel_size=3, padding='same')(main)
	main = BatchNormalization()(main)
	main = Activation('relu')(main)
	main = Conv1D(filters=256, kernel_size=3, padding='same')(main)
	main = concatenate([main, i_l1],axis=1)
	main = BatchNormalization()(main)
	main = Activation('relu')(main)
	main = MaxPooling1D(pool_size=2)(main)
	i_l1 = Conv1D(filters=512, kernel_size=1, padding='same')(main)

	main = Conv1D(filters=512, kernel_size=3, padding='same')(main)
	main = BatchNormalization()(main)
	main = Activation('relu')(main)
	main = Conv1D(filters=512, kernel_size=3, padding='same')(main)
	main = concatenate([main, i_l1],axis=1)
	main = BatchNormalization()(main)
	main = Activation('relu')(main)
	main = MaxPooling1D(pool_size=2)(main)

	main = GlobalMaxPool1D()(main)

	main = Dense(1024, activation='relu')(main)  # Should be 4096
	main = Dense(512, activation='relu')(main)  # Should be 2048

	out1 = Dense(classes_1,activation='softmax')(main)
	out2 = Dense(classes_2,activation='softmax')(main)
	out3 = Dense(classes_3,activation='softmax')(main)
	out4 = Dense(classes_4,activation='softmax')(main)
	out5 = Dense(classes_5,activation='softmax')(main)
	out6 = Dense(classes_6,activation='softmax')(main)

	model = Model(inputs=[inp], outputs=[out1,out2,out3,out4,out5,out6])
	optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
	model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
	return model


def build_mlp(Traniable_embedding,embedding_matrix,max_len,kmer_size,classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
	inp = Input(shape=(max_len,),dtype='uint16')
	max_features = 4**kmer_size +1

	if Traniable_embedding ==True:
		main = Embedding(max_features, 128)(inp)
	else:
		main = Embedding(max_features, vector_size, weights=[embedding_matrix],trainable=False)(inp)

	main = Dense(512)(main)
	main = GlobalMaxPool1D()(main)
	main = Dropout(0.25)(main)
	main = Dense(512)(main)  
	main = Dropout(0.25)(main)
	out1 = Dense(classes_1,activation='softmax')(main)
	out2 = Dense(classes_2,activation='softmax')(main)
	out3 = Dense(classes_3,activation='softmax')(main)
	out4 = Dense(classes_4,activation='softmax')(main)
	out5 = Dense(classes_5,activation='softmax')(main)
	out6 = Dense(classes_6,activation='softmax')(main)
	model = Model(inputs=[inp], outputs=[out1,out2,out3,out4,out5,out6])
	optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
	model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
	return model
