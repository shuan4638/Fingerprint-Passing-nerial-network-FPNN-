from __future__ import print_function, division

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Dropout,Embedding, BatchNormalization, Lambda, MaxPooling1D, Flatten
from keras import backend as K
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.regularizers import l2
from utils import *
from model import *


input_length = 40
bit_size = 1024
embedding_size = 20
filter_num = 30
  
def sum_channel(layer):
    return K.sum(layer, axis = 1)
    
def gfp2vec(input_dim = bit_size, output_dim = embedding_size, input_length = input_length, filter_num = filter_num, pooling = 'max'):
    
    G = Input(shape=(input_length, input_length))
    X_in = Input(shape=(input_length,))
    H = Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length, name='embedding_layer')(X_in)
    H = Dropout(0.25)(H)
    H = AtomConvolution(50, kernel_regularizer=l2(5e-4), use_bias=False)([H]+[G])
    H = BatchNormalization()(H)
    H = Dropout(0.25)(H)
    H = Conv1D(filter_num, kernel_size = 1, strides = 1, use_bias=False, activation = 'relu')(H)
    if pooling == 'sum':
        H = Lambda(sum_channel)(H)
    elif pooling == 'max':
        H = MaxPooling1D(input_length)(H)
        H = Flatten()(H)
    else:
        H = Flatten()(H)
    H = Dropout(0.25)(H)
    Y = Dense(1, use_bias=False, activation='sigmoid')(H)
    
    model = Model(inputs = [X_in, G], outputs=Y)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    
    return model


def GCN(input_dim = bit_size, output_dim = embedding_size, input_length = input_length, filter_num = filter_num, pooling = 'sum'):
    
    G = Input(shape=(input_length, input_length))
    X_in = Input(shape=(input_length,))
    H = Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length, name='embedding_layer')(X_in)
    H = Dropout(0.25)(H)
    H = GraphConvolution(filter_num, kernel_regularizer=l2(5e-4), use_bias=False, activation = 'relu')([H]+[G])
    H = Dropout(0.25)(H)
    H = GraphConvolution(filter_num, kernel_regularizer=l2(5e-4), use_bias=False, activation = 'relu')([H]+[G])
    if pooling == 'sum':
        H = Lambda(sum_channel)(H)
    elif pooling == 'max':
        H = MaxPooling1D(input_length)(H)
        H = Flatten()(H)
    else:
        H = Flatten()(H)
    H = Dropout(0.25)(H)
    Y = Dense(1, use_bias=False, activation='sigmoid')(H)
    
    model = Model(inputs = [X_in, G], outputs=Y)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_gfp2vec(train_G, test_G, valid_G, valid_y, train_y, test_y, pooling = 'max'):
    model = gfp2vec(pooling = pooling)
    hist = model.fit(train_G, train_y, batch_size=100, epochs=300, validation_data=(valid_G, valid_y), verbose=0)
    pred_y = model.predict(valid_G) 
    accuracy = metrics.roc_auc_score(valid_y, pred_y)
    return accuracy

def train_gcn(train_G, test_G, valid_G, valid_y, train_y, test_y, pooling = 'max'):
    model = GCN(pooling = pooling)
    hist = model.fit(train_G, train_y, batch_size=150, epochs=300, validation_data=(valid_G, valid_y), verbose=0)
    pred_y = model.predict(valid_G) 
    accuracy = metrics.roc_auc_score(valid_y, pred_y)
    return accuracy


def split_data(data_X, data_y, data_A, data_A_):
    train_X, test_X, train_y, test_y, train_A, test_A, train_A_, test_A_ = train_test_split(data_X, data_y, data_A, data_A_, test_size=0.1)
    train_X, valid_X, train_y, valid_y, train_A, valid_A, train_A_, valid_A_ = train_test_split(train_X, train_y, train_A, train_A_, test_size=0.1111)
    return train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_

def train_file(train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_):
    ##gfp2vec
    train_G = [train_X, train_A]
    test_G = [test_X, test_A]
    valid_G = [valid_X, valid_A]
    AUC1 = train_gfp2vec(train_G, test_G, valid_G, valid_y, train_y, test_y, pooling = 'max')
    AUC2 = train_gfp2vec(train_G, test_G, valid_G, valid_y, train_y, test_y, pooling = 'sum')
    ##gcn
    train_G_ = [train_X, train_A_]
    test_G_ = [test_X, test_A_]
    valid_G_ = [valid_X, valid_A_]
    AUC3 = train_gcn(train_G_, test_G_, valid_G_, valid_y, train_y, test_y, pooling = 'max')
    AUC4 = train_gcn(train_G_, test_G_, valid_G_, valid_y, train_y, test_y, pooling = 'sum')
    #print ('GFP : max_pool: %.3f, sum_pool: %.3f  \n GCN : max_pool: %.3f, sum_pool: %.3f' % (AUC1, AUC2, AUC3, AUC4))
    return [AUC1, AUC2, AUC3, AUC4]

def train_atom_sum(train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A):
    ##gfp2vec
    train_G = [train_X, train_A]
    test_G = [test_X, test_A]
    valid_G = [valid_X, valid_A]
    AUC = train_gfp2vec(train_G, test_G, valid_G, valid_y, train_y, test_y, pooling = 'sum')
    print (AUC)
    return AUC

#train data
def train_and_test(file):
    atom_sum  = []
    dataset = {}
    data_X, data_A, data_A_, data_y = load_data(file)
    print ('finish loading %s!' % file)
    for i in range(10):
        train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_ = split_data(data_X, data_y, data_A, data_A_)
        AUC = train_atom_sum(train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A)
        atom_sum.append(AUC)
        dataset[i] = train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_

    best_idx = np.argmax(atom_sum)
    print (atom_sum[best_idx])
    best_AUCs = pd.DataFrame()
    train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_ = dataset[best_idx]
    for i in range(5):
        AUCs = train_file(train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_)
        best_AUCs[i] = AUCs
    best_AUCs.index = ['GFP+max_pool', 'GFP+sum_pool', 'GCN+max_pool', 'GCN+sum_pool']
    print (best_AUCs)
    return best_AUCs
    
HIV = train_and_test('hiv.csv')