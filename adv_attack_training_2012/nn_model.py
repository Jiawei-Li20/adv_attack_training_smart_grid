#build the NN models: RNN module, NN module
import tensorflow
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Layer, Dense, Dropout, Activation, Flatten
from tensorflow.compat.v1.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.compat.v1.keras.layers import LSTM, Embedding,SimpleRNN
# import np_utils
from tensorflow.python.platform import flags
from numpy import shape
import numpy as np
#from skimage import io, color, exposure, transform
import os
import glob
import h5py
import pandas as pd
import numpy as np
from scipy.io import loadmat

class PiecewiseLinearFunc(Layer):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        # prepare func
        m=loadmat("QuantileSolutionRampStatic161028.mat")
        capacity=np.array(m['capacity']).reshape(-1)
        capacity=np.rint(capacity).astype(np.int64)
        mc=np.array(m['mc']).reshape(-1)
        id_=np.argsort(mc)
        capacity=capacity[id_]
        mc = mc[id_]
        bound_capacity = [0]
        intercepts, slope = [0], [0]
        mean = 4.16241129e+04
        var = 1.60763191e+08
        LARGE_num = 2e4
        x, y = 0,0
        for i in range(0,len(mc)):            
            intercepts.append(y - mc[i]*x)
            slope.append(mc[i])
            x += capacity[i]
            y += capacity[i] * mc[i]
            bound_capacity.append(x)

        # intercepts.append(y - mc[-1]*x)
        # slope.append(mc[-1])
        # x += (LARGE_num + capacity[-1])
        # y += (LARGE_num + capacity[-1])* mc[i]
        # bound_capacity.append(x)            

        bound_capacity = (np.array(bound_capacity[:-1]) - mean)/np.sqrt(var)
        intercepts = np.array(intercepts) + np.array(slope) * mean        
        slope = np.array(slope) * np.sqrt(var)
        print(bound_capacity, intercepts, slope)
        self.boundaries = tf.Variable(tf.constant(bound_capacity.astype(np.float32)), trainable=False)
        self.slopes = tf.Variable(tf.constant(slope.astype(np.float32)), trainable=False)
        self.intercepts = tf.Variable(tf.constant(intercepts.astype(np.float32)), trainable=False)

    def call(self, inputs):
        # batch_size = tf.shape(inputs)[0]
        # print(tf.shape(inputs))
        x = inputs

        # Compute the index of the segment that each x value falls in
        segment_idx = tf.searchsorted(self.boundaries, x, side='left')
        # print(segment_idx)
        # Compute the slopes and intercepts for each segment
        slopes = tf.gather(self.slopes, segment_idx)
        intercepts = tf.gather(self.intercepts, segment_idx)

        # Compute the predicted y values for each x value
        y_pred = slopes * x + intercepts
        return y_pred

def rnn_piecewise_model(seq_length, input_dim, output_dim):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(seq_length, input_dim), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    #model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    #model.add(Activation('relu'))
    model.add(Dense(32))
    #model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Dense(output_dim))
    model.add(Activation('linear'))
    model.add(PiecewiseLinearFunc())
    return model

if __name__ == '__main__':
    mean = 4.16241129e+04
    var = 1.60763191e+08
    ol = np.array([10000,20000,30000,80000])
    l = (ol - mean)/np.sqrt(var)
    test = tf.convert_to_tensor(l, dtype=tf.float32)
    fun = PiecewiseLinearFunc()
    with tf.Session() as sess:
        print(fun.call(test).eval())
        print(plant_cost_curve[ol])

def rnn_model(seq_length, input_dim, output_dim):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(seq_length, input_dim), return_sequences=False, dtype=tf.float64))
    model.add(Dropout(0.2, dtype=tf.float64))
    model.add(Dense(32, dtype=tf.float64))
    #model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2, dtype=tf.float64))
    #model.add(Activation('relu'))
    model.add(Dense(32, dtype=tf.float64))
    #model.add(Activation('relu'))
    model.add(Dense(16, dtype=tf.float64))
    model.add(Dense(output_dim, dtype=tf.float64))
    model.add(Activation('linear', dtype=tf.float64))
    return model

def nn_model(input_dim, output_dim):
    model=Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Dense(output_dim, init='normal'))
    model.add(Activation('linear'))

    return model


def lstm_model(seq_length, input_dim, output_dim):
    model = Sequential()
    #model.add(SimpleRNN(64, input_shape=(seq_length, input_dim), return_sequences=False))
    model.add(LSTM(64, input_shape=(seq_length, input_dim), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    #model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    #model.add(Activation('relu'))
    model.add(Dense(32))
    #model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Dense(output_dim, init='normal'))
    model.add(Activation('linear'))
    return model


def svm_model(seq_length, input_dim):

    return model
