from __future__ import print_function
import numpy as np
import pandas as pd 
from numpy import array
from numpy.random import randint

import keras
import random
import tensorflow as tf

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score, accuracy_score
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Concatenate, LeakyReLU, concatenate,GRU, Bidirectional, MaxPool1D,GlobalMaxPool1D,add
from keras.layers import Dense, Embedding, Input, Masking, Dropout, MaxPooling1D,Lambda, BatchNormalization, Reshape
from keras.layers import LSTM, TimeDistributed, AveragePooling1D, Flatten,Activation,ZeroPadding1D,SeparableConv1D, GlobalAveragePooling1D
from keras.optimizers import Adam, rmsprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint, CSVLogger
from keras.layers import Conv1D, GlobalMaxPooling1D, ConvLSTM2D, Bidirectional,RepeatVector
from keras.regularizers import *
from keras import regularizers
from keras.layers import concatenate as concatLayer
from keras.utils import plot_model
import os
#from ._conv import register_converters as _register_converters


#import keras_metrics

import itertools
from itertools import product

from gensim.models import KeyedVectors

vector_size = 128
max_features = (4**6) +1
def get_word_to_int(kmer_size):
	bases=['1','2','3','4']
	all_kmers = [''.join(p) for p in itertools.product(bases, repeat=kmer_size)]
	word_to_int = dict()
	word_to_int = word_to_int.fromkeys(all_kmers)
	keys = range(1,len(all_kmers)+1)
	for k in keys:
	    word_to_int[all_kmers[k-1]] = keys[k-1]
	return word_to_int

# Creating embedding matrix from the word2vec model
def build_embedding_matrix(model_path,word_to_int):
    word2vec = KeyedVectors.load_word2vec_format(model_path)
    kmer_size = 6
    max_features = 4**kmer_size + 1
    embedding_matrix = np.zeros((max_features,vector_size))
    for i in word_to_int.values():
        embedding_matrix[i,:] = word2vec.wv[str(i)]
    embedding_matrix = Embedding(max_features, vector_size, weights=[embedding_matrix],trainable=False)
    #embedding_matrix = word2vec.wv.get_keras_embedding(train_embeddings=False)
    return embedding_matrix

# Convert the sequence of chracters to sequence of kmers
def oneHotEncoding_to_kmers(encoded_list,kmer_size,word_to_int):
    word_list = []
    ch_str = str(encoded_list.tolist()).replace(',','').replace('[','').replace(']','').replace(' ','')
    for i in range(len(ch_str) - kmer_size + 1):
        word_list.append(int(word_to_int[ch_str[ i : i + kmer_size ]]))
    return word_list

# Convert the sequence of chracters to sequence of kmers
def oneHotEncoding_to_kmers_csv(encoded_list,kmer_size,word_to_int):
    word_list = []
    ch_str = str(encoded_list).replace('\n','').replace('[','').replace(']','').replace(' ','')
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
def simulate_ngs_generator(df,len1 ,batch_size,max_len,kmer_size,save_mem):
    #if classes_6 != 0:
    #    y_sim_6 = df['species-'].values
    word_to_int = get_word_to_int(kmer_size)
    if save_mem == False:
        y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
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
            #if classes_6 != 0:
            #    y_6 = y_sim_6[batch_size*counter:batch_size*(counter+1)].astype('uint16')
            counter += 1
            if counter ==number_of_batches:
                counter = 0
                del x_sim,  y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5
                y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
                #if classes_6 != 0:
                #    y_sim_6 = df['species-'].values

                x_sim = df['encoded'].apply(lambda x : simulate_reads_range(x,min_len=len1))
                x_sim = pad_sequences(x_sim.values,maxlen=max_len)
                x_sim = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
            yield X_batch, [y_1,y_2,y_3,y_4,y_5]
    
    else:
        path = df
        #samples_per_epoch = sum(1 for line in open(path))
        df = pd.read_csv(path)#,chunksize=[batch_size*counter:batch_size*(counter+1)])
        df['len'] = df['encoded'].apply(lambda x: len(x))
        df = df.sample(frac=1).reset_index(drop=True)
        df = df[df['len']>len1]
        samples_per_epoch = df.shape[0]
        y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
    
        number_of_batches = samples_per_epoch//batch_size
        counter=0

        while counter <=number_of_batches :
            # many lines were inteded to load from the disk rather saving memory
            #df = pd.read_csv(path,chunksize=[batch_size*counter:batch_size*(counter+1)])
            #y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
            df2 = df.iloc[batch_size*counter:batch_size*(counter+1),:]
            df2['encoded_'] = df2['encoded'].apply(lambda x: oneHotEncoding_to_kmers_csv(encoded_list=x,
            	kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()
            x_sim = df2['encoded_'].apply(lambda x : simulate_reads_range(x,min_len=len1))
            x_sim = pad_sequences(x_sim.values,maxlen=max_len)
            X_batch = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
            y_1 = y_sim_1[batch_size*counter:batch_size*(counter+1)].astype('uint8')
            y_2 = y_sim_2[batch_size*counter:batch_size*(counter+1)].astype('uint8')
            y_3 = y_sim_3[batch_size*counter:batch_size*(counter+1)].astype('uint8')
            y_4 = y_sim_4[batch_size*counter:batch_size*(counter+1)].astype('uint16')
            y_5 = y_sim_5[batch_size*counter:batch_size*(counter+1)].astype('uint16')
            #if classes_6 != 0:
            #    y_6 = y_sim_6[batch_size*counter:batch_size*(counter+1)].astype('uint16')
            counter += 1
            if counter ==number_of_batches:
                counter = 0
                del x_sim,df2,X_batch
                #if cl560475asses_6 != 0:38/
                #    y_sim_6 = df['species-'].values
                #df = pd.read_csv(path,chunksize=[batch_size*counter:batch_size*(counter+1)])
                #y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
                df2 = df.iloc[batch_size*counter:batch_size*(counter+1),:]
                df2['encoded_'] = df2['encoded'].apply(lambda x: oneHotEncoding_to_kmers_csv(encoded_list=x,
                	kmer_size=kmer_size,word_to_int = word_to_int)).values.tolist()
                x_sim = df2['encoded_'].apply(lambda x : simulate_reads_range(x,min_len=len1))
                x_sim = pad_sequences(x_sim.values,maxlen=max_len)
                X_batch = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
                y_1 = y_sim_1[batch_size*counter:batch_size*(counter+1)].astype('uint8')
                y_2 = y_sim_2[batch_size*counter:batch_size*(counter+1)].astype('uint8')
                y_3 = y_sim_3[batch_size*counter:batch_size*(counter+1)].astype('uint8')
                y_4 = y_sim_4[batch_size*counter:batch_size*(counter+1)].astype('uint16')
                y_5 = y_sim_5[batch_size*counter:batch_size*(counter+1)].astype('uint16')
            yield X_batch, [y_1,y_2,y_3,y_4,y_5]

# A generator function of a fixed length subsequence that changes in each epoch
def simulate_ngs_generator_fixed_len(df,len1,batch_size,max_len):
    y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
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
        #y_6 = y_sim_6[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        counter += 1
        if counter ==number_of_batches:
            counter = 0
            del x_sim
            y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
            x_sim = df['encoded'].apply(lambda x : simulate_reads(x,len1=len1))
            x_sim = pad_sequences(x_sim.values,maxlen=max_len)
            x_sim = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
        yield X_batch,[y_1,y_2,y_3,y_4,y_5]

def simulate_multi_task_generator(df,len1 ,batch_size,max_len,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
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
        #y_6 = y_sim_6[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        y_1 = to_categorical(y_1,num_classes=classes_1)
        y_2 = to_categorical(y_2,num_classes=classes_2)
        y_3 = to_categorical(y_3,num_classes=classes_3)
        y_4 = to_categorical(y_4,num_classes=classes_4)
        y_5 = to_categorical(y_5,num_classes=classes_5)
        #y_6 = to_categorical(y_6,num_classes=classes_6)
        Y = np.hstack((y_1,y_2,y_3,y_4,y_5))
        counter += 1
        if counter ==number_of_batches:
            counter = 0
            del x_sim,  y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5
            y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
            x_sim = df['encoded'].apply(lambda x : simulate_reads_range(x,min_len=len1))
            x_sim = pad_sequences(x_sim.values,maxlen=max_len)
            x_sim = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
        yield X_batch, Y

# A generator function of a fixed length subsequence that changes in each epoch
def simulate_multi_task_generator_fixed_len(df,len1,batch_size,max_len,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
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
        #y_6 = y_sim_6[batch_size*counter:batch_size*(counter+1)].astype('uint16')
        y_1 = to_categorical(y_1,num_classes=classes_1)
        y_2 = to_categorical(y_2,num_classes=classes_2)
        y_3 = to_categorical(y_3,num_classes=classes_3)
        y_4 = to_categorical(y_4,num_classes=classes_4)
        y_5 = to_categorical(y_5,num_classes=classes_5)
        #y_6 = to_categorical(y_6,num_classes=classes_6)
        Y = np.hstack((y_1,y_2,y_3,y_4,y_5))
        counter += 1
        if counter ==number_of_batches:
            counter = 0
            del x_sim
            y_sim_1,y_sim_2,y_sim_3,y_sim_4,y_sim_5 = df['phylum'].values,df['class_'].values,df['order'].values,df['family'].values,df['genus'].values
            x_sim = df['encoded'].apply(lambda x : simulate_reads(x,len1=len1))
            x_sim = pad_sequences(x_sim.values,maxlen=max_len)
            x_sim = array(np.concatenate(x_sim).reshape(x_sim.shape[0],max_len).tolist()).astype('uint16')
        yield X_batch,Y

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

from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# F1 callback on the end on each epoch
class f1_callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.f1s.append(sklm.f1_score(targ, predict,average='micro'))
        self.confusion.append(sklm.confusion_matrix(targ.argmax(axis=1),predict.argmax(axis=1)))

        return


def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

# A function for building ResNet model
def build_resnet(Traniable_embedding,embedding_matrix,max_len,kmer_size,metrics,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):

    inp = Input(shape=(max_len,),dtype='uint16')
    max_features = 4**kmer_size +1
    if Traniable_embedding ==True:
        main = Embedding(5, 128)(inp)
    else:
        #main = Embedding(max_features, vector_size, weights=[embedding_matrix],trainable=False)(inp)
        main = embedding_matrix(inp)

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
    if classes_6 !=0:
        out6 = Dense(classes_6,activation='softmax')(main)
    else:
        pass
    if metrics =='f1':
        metrics = f1
    elif metrics == 'precision-recall':
        metrics = [keras_metrics.precision(), keras_metrics.recall()]
    else:
        pass
    model = Model(inputs=[inp], outputs=[out1,out2,out3,out4,out5])
    optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=[metrics])
    return model


def build_mlp(Traniable_embedding,embedding_matrix,max_len,kmer_size,metrics,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    inp = Input(shape=(max_len,),dtype='uint16')
    max_features = 4**kmer_size +1

    if Traniable_embedding ==True:
        main = Embedding(max_features, 128)(inp)
    else:
        #main = Embedding(max_features, vector_size, weights=[embedding_matrix],trainable=False)(inp)
        main = embedding_matrix(inp)
    
    main = Dense(512)(main)
    #main = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,share_weights=True)(main)
    #main = Flatten()(main)
    main = GlobalMaxPool1D()(main)
    main = Dropout(0.25)(main)
    main = Dense(512)(main)  
    main = Dropout(0.25)(main)
    out1 = Dense(classes_1,activation='softmax')(main)
    out2 = Dense(classes_2,activation='softmax')(main)
    out3 = Dense(classes_3,activation='softmax')(main)
    out4 = Dense(classes_4,activation='softmax')(main)
    out5 = Dense(classes_5,activation='softmax')(main)
    #out2 = hierarichal_softmax(classes_2,out1,'class_',tree)(out2)
    #out3 = hierarichal_softmax(classes_3,out2,'order',tree)(out3)
    #out4 = hierarichal_softmax(classes_4,out3,'family',tree)(out4)
    #out5 = hierarichal_softmax(classes_5,out4,'genus',tree)(out5)

    if classes_6 !=0:
        out6 = Dense(classes_6,activation='softmax')(main)
    else:
        pass

    if metrics =='f1':
        metrics = f1
    elif metrics == 'precision-recall':
        metrics = [keras_metrics.precision(), keras_metrics.recall()]
    else:
        pass
    model = Model(inputs=[inp], outputs=[out1,out2,out3,out4,out5])
    optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=[metrics])
    return model


def build_gru(Traniable_embedding,embedding_matrix,max_len,kmer_size,metrics,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    inp = Input(shape=(max_len,),dtype='uint16')
    max_features = 4**kmer_size +1

    if Traniable_embedding ==True:
        main = Embedding(max_features, 128)(inp)
    else:
        #main = Embedding(max_features, vector_size, weights=[embedding_matrix],trainable=False)(inp)
        main = embedding_matrix(inp)
    
    main = Bidirectional(GRU(16, return_sequences=True))(main)
    #main = AttentionDecoder(8, 5)(main)
    main = Attention_layer()(main)
    main = Dense(64)(main)
    #main = GlobalMaxPool1D()(main)
    main = Dropout(0.25)(main)
    main = Dense(128)(main)  
    main = Dropout(0.25)(main)
    out1 = Dense(classes_1,activation='softmax')(main)
    out2 = Dense(classes_2,activation='softmax')(main)
    out3 = Dense(classes_3,activation='softmax')(main)
    out4 = Dense(classes_4,activation='softmax')(main)
    out5 = Dense(classes_5,activation='softmax')(main)
    
    if classes_6 !=0:
        out6 = Dense(classes_6,activation='softmax')(main)
    else:
        pass

    if metrics =='f1':
        metrics = f1
    elif metrics == 'precision-recall':
        metrics = [keras_metrics.precision(), keras_metrics.recall()]
    else:
        pass
    model = Model(inputs=[inp], outputs=[out1,out2,out3,out4,out5])
    optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=[metrics])
    return model

def build_multitask_mlp(Traniable_embedding,embedding_matrix,max_len,kmer_size,metrics,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    inp = Input(shape=(max_len,),dtype='uint16')
    max_features = 4**kmer_size +1
    all_classes = classes_1+classes_2+classes_3+classes_4+classes_5+classes_6
    if Traniable_embedding ==True:
        main = Embedding(max_features, 128)(inp)
    else:
        #main = Embedding(max_features, vector_size, weights=[embedding_matrix],trainable=False)(inp)
        main = embedding_matrix(inp)

    main = Dense(512)(main)
    main = GlobalMaxPool1D()(main)
    main = Dropout(0.25)(main)
    main = Dense(512)(main)  
    main = Dropout(0.25)(main)
    output = Dense(all_classes)(main)
    output = Activation('sigmoid')(output)
    metrics = 'mse'
    model = Model(inputs=[inp], outputs=[output])
    optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
    model.compile(loss=multitask_loss,optimizer=optimizer,metrics=[metrics])
    return model

def build_tpu_mlp(Traniable_embedding,embedding_matrix,max_len,kmer_size,metrics,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):
    from tensorflow.python.keras.layers import Input, GlobalMaxPool1D, Dense, Embedding,Dropout
    inp = Input(shape=(max_len,),dtype='int32')
    max_features = 4**kmer_size +1
    main = Embedding(max_features, 128)(inp)
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
        
    model_ = tf.keras.Model(inputs=[inp], outputs=[out1,out2,out3,out4,out5])
    #optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4, epsilon=1e-8, decay=0.)
    #model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=[metrics])
    model_.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['acc'])
    # This address identifies the TPU we'll use when configuring TensorFlow.
    TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    tf.logging.set_verbosity(tf.logging.INFO)

    model = tf.contrib.tpu.keras_to_tpu_model(
        model_,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

    return model

## Attention Implementation in Keras by https://github.com/BITDM/bitdm.github.io
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

class Attention_layer(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(Attention_layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        a = K.exp(uit)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


# Building a seperable con simialr to Google seq2species architecture
# These parameters were tune on RefSeq amplicon data not insiloc processed SILVA, but it's the closest way to compare AmpliconNet to Seq2Species
def build_sepconv(Traniable_embedding,embedding_matrix,max_len,kmer_size,metrics,
    classes_1,classes_2,classes_3,classes_4,classes_5,classes_6):

    inp = Input(shape=(max_len,),dtype='uint16')
    max_features = 4**kmer_size +1
    if Traniable_embedding ==True:
        main = Embedding(max_features, 128)(inp)
    else:
        #main = Embedding(max_features, vector_size, weights=[embedding_matrix],trainable=False)(inp)
        main = embedding_matrix(inp)

    main = SeparableConv1D(84 ,(5))(main)
    main = Dropout(0.5)(main)
    main = LeakyReLU()(main)

    main = SeparableConv1D(58,(9))(main)
    main = Dropout(0.5)(main)
    main = LeakyReLU()(main)

    main = SeparableConv1D(180,(13),)(main)
    main = Dropout(0.5)(main)
    main = LeakyReLU()(main)

    main = Dense(2800)(main)
    main = Dropout(0.5)(main)
    main = LeakyReLU()(main)
    #main = Dense(2800)(main)

    main = GlobalAveragePooling1D()(main)
    out1 = Dense(classes_1,activation='softmax')(main)
    out2 = Dense(classes_2,activation='softmax')(main)
    out3 = Dense(classes_3,activation='softmax')(main)
    out4 = Dense(classes_4,activation='softmax')(main)
    out5 = Dense(classes_5,activation='softmax')(main)
    if classes_6 !=0:
        out6 = Dense(classes_6,activation='softmax')(main)
    else:
        pass
    if metrics =='f1':
        metrics = f1
    elif metrics == 'precision-recall':
        metrics = [keras_metrics.precision(), keras_metrics.recall()]
    else:
        pass
    model = Model(inputs=[inp], outputs=[out1,out2,out3,out4,out5])
    optimizer = Adam(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=[metrics])
    return model

# Hierarchical Softmax layer for training
# Under develeopment
from keras.layers import Lambda
#tree = pd.read_csv('./V2_dumpy/tree.csv')
ranks = ['phylum','class_','order','family','genus']
def hierarichal_softmax(num_class,higher_rank_softmax,rank,tree, **kwargs):
    #x = Dense(num_class,'softmax')(x)
    def func(x):
        
        index = ranks.index(rank)
        higher_rank = ranks[index-1]
        y_1 = K.argmax(higher_rank_softmax,axis=-1)
        pool = K.equal(tree[higher_rank],y_1[0])
        print(pool)

        y_2_ind = tree[pool][rank].unique()
        
        y_2 = x[y_2_ind].argmax(-1)
        y_2 = y_2_ind[y_2]
        y_2 = to_categorical(y_2,num_classes=num_class)
        
        return y_2
        
    return Lambda(func, **kwargs)

# Capsule Layer

gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.3
rate_drop_dense = 0.3

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
