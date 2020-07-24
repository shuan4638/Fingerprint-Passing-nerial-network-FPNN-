import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Dropout,Embedding, Lambda, MaxPooling1D, Flatten
from keras import backend as K
from __future__ import print_function, division
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.regularizers import l2
from utils import *
from model import *

input_length = 40
bit_size = 1024
embedding_size = 50
filter_num = 30
  
def sum_channel(layer):
    return K.sum(layer, axis = 1)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
    
def AGCN(input_dim = bit_size, output_dim = embedding_size, input_length = input_length, filter_num = filter_num, pooling = 'sum'):
    
    G = Input(shape=(input_length, input_length))
    X_in = Input(shape=(input_length,))
    H = Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length, name='embedding_layer')(X_in)
    H = Dropout(0.25)(H)
    H = AtomConvolution(50, kernel_regularizer=l2(5e-4), use_bias=False)([H]+[G])
    H = Dropout(0.25)(H)
    H = Conv1D(filter_num, kernel_size = 1, strides = 1, use_bias=False)(H)
    if pooling == 'sum':
        H = Lambda(sum_channel)(H)
    elif pooling == 'max':
        H = MaxPooling1D(input_length)(H)
        H = Flatten()(H)
    else:
        H = Flatten()(H)
    H = Dropout(0.25)(H)
    Y = Dense(1)(H)
    
    model = Model(inputs = [X_in, G], outputs=Y)
    model.compile('adam', 'mse', metrics=['mean_absolute_error', 'mean_squared_error'])
        
    return model


def GCN(input_dim = bit_size, output_dim = embedding_size, input_length = input_length, filter_num = filter_num, pooling = 'sum'):
    
    G = Input(shape=(input_length, input_length))
    X_in = Input(shape=(input_length,))
    H = Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length, name='embedding_layer')(X_in)
    H = Dropout(0.25)(H)
    H = GraphConvolution(filter_num, kernel_regularizer=l2(5e-4), use_bias=False, activation = 'relu')([H]+[G])
    H = Dropout(0.25)(H)
    H = GraphConvolution(filter_num, kernel_regularizer=l2(5e-4), use_bias=False, activation = 'tanh')([H]+[G])
    if pooling == 'sum':
        H = Lambda(sum_channel)(H)
    elif pooling == 'max':
        H = MaxPooling1D(input_length)(H)
        H = Flatten()(H)
    else:
        H = Flatten()(H)
    H = Dropout(0.25)(H)
    Y = Dense(1)(H)
    
    model = Model(inputs = [X_in, G], outputs=Y)
    model.compile('adam', 'mse', metrics=['mean_absolute_error', 'mean_squared_error'])
    
    return model


def train_agcn(train_G, test_G, train_y, test_y, valid_G, valid_y, pooling = 'max'):
    model = AGCN(pooling = pooling)
    hist = model.fit(train_G, train_y, batch_size=150, epochs=300, validation_data=(valid_G, valid_y), verbose=0)
    loss, mae, mse = model.evaluate(valid_G, valid_y, verbose=0)
    return [loss, mae, mse]

def train_gcn(train_G, test_G, train_y, test_y, valid_G, valid_y, pooling = 'max'):
    model = GCN(pooling = pooling)
    hist = model.fit(train_G, train_y, batch_size=150, epochs=300, validation_data=(valid_G, valid_y), verbose=0)
    loss, mae, mse = model.evaluate(valid_G, valid_y, verbose=0)
    return [loss, mae, mse]

def split_data(data_X, data_y, data_A, data_A_):
    train_X, test_X, train_y, test_y, train_A, test_A, train_A_, test_A_ = train_test_split(data_X, data_y, data_A, data_A_, test_size=0.1)
    train_X, valid_X, train_y, valid_y, train_A, valid_A, train_A_, valid_A_ = train_test_split(train_X, train_y, train_A, train_A_, test_size=0.1111)
    return train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_

def train_atom_sum(train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A):
    ##agcn
    train_G = [train_X, train_A]
    test_G = [test_X, test_A]
    valid_G = [valid_X, valid_A]
    result = train_agcn(train_G, test_G, train_y, test_y, valid_G, valid_y, pooling = 'sum')
    mse = result[2]
    return mse

def train_file(train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_):
    ##agcn
    train_G = [train_X, train_A]
    test_G = [test_X, test_A]
    valid_G = [valid_X, valid_A]
    result1_1 = train_agcn(train_G, test_G, train_y, test_y, valid_G, valid_y)
    result1_2 = train_agcn(train_G, test_G, train_y, test_y, valid_G, valid_y, pooling = 'sum')
    ##gcn
    train_G_ = [train_X, train_A_]
    test_G_ = [test_X, test_A_]
    valid_G_ = [valid_X, valid_A_]
    result2_1 = train_gcn(train_G_, test_G_, train_y, test_y, valid_G_, valid_y)
    result2_2 = train_gcn(train_G, test_G, train_y, test_y, valid_G, valid_y, pooling = 'sum')
    print ('GFP : max_pool: %s, sum_pool: %s  \n GCN : max_pool: %s, sum_pool: %s' % (result1_1, result1_2, result2_1, result2_2))
    mses = [result1_1[2], result1_2[2], result2_1[2], result2_2[2]]

    return mses


def test_on_best_r(file):
    atom_sum  = []
    dataset = {}
    data_X, data_A, data_A_, data_y = load_data(file)
    print ('finish loading %s!' % file)
    for i in range(10):
        train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_ = split_data(data_X, data_y, data_A, data_A_)
        mse = train_atom_sum_r(train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A)
        atom_sum.append(mse)
        dataset[i] = train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_

    best_idx = np.argmin(atom_sum)
    print (atom_sum[best_idx])
    best_mses = pd.DataFrame()
    train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_ = dataset[best_idx]
    for i in range(5):
        mses = train_file_r(train_X, test_X, valid_X, train_y, test_y, valid_y, train_A, test_A, valid_A, train_A_, test_A_, valid_A_)
        best_mses[i] = mses
    best_mses.index = ['GFP+max_pool', 'GFP+sum_pool', 'GCN+max_pool', 'GCN+sum_pool']
    print (best_mses)
    return best_mses

ESOL = test_on_best_r('ESOL.csv')
