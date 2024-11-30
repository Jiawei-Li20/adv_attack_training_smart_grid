import os, sys, math, itertools, csv
import random
random.seed(11)

import datetime as dt
import time as t
import pandas as pd
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt

import scipy.stats as stats
from statsmodels.tsa import stattools

import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import keras
from keras.optimizers import SGD

from datasets.data import *
from util import reorganize, reorganize2, calculate_mae
from nn_model import rnn_model, nn_model, svm_model


def scaled_gradient(x, predictions, target):
    loss = tf.square(predictions - target)
    grad, = tf.gradients(loss, x)
    signed_grad = tf.sign(grad)
    return grad, signed_grad


def check_constraint(x_orig, x_new, bound):
    for i in range(24):#?
        j=0
        x_new[:, i, j] = np.clip(x_new[:,i,j],
                                    x_orig[:, i, j] - bound * np.abs(x_orig[:, i, j]),
                                    x_orig[:, i, j] + bound * np.abs(x_orig[:, i, j]))
        j=1
        x_new[:, i, j] = np.clip(x_new[:,i,j],
                                    x_orig[:, i, j] - 5*bound * np.abs(x_orig[:, i, j]),
                                    x_orig[:, i, j] + 5*bound * np.abs(x_orig[:, i, j]))


    return x_new

def bound_val(x_orig, bound):
    x_low=[]
    x_high=[]
    for i in range(np.shape(x_orig)[0]):
        x_low.append(x_orig[i,0,0]-bound*np.abs(x_orig[i, 0, 0]))
        x_high.append(x_orig[i, 0, 0] + bound * np.abs(x_orig[i, 0, 0]))
    x_low=np.array(x_low, dtype=float)
    x_high=np.array(x_high, dtype=float)

    return x_low, x_high

def temp_bound(adv, orig, temp_val):
    for i in range(1, 2):
        for j in range(len(adv)):
            adv[j, i]=np.clip(adv[j,i], orig[j,i]-temp_val, orig[j,i]+temp_val)
    return adv

# ==============================
# Main Settings and Initialization
# ==============================
print('=' * 30 + '\nInitialization begins.\n' + '=' * 30)
features = ['actual', 'calendar', 'weather']  # features to be loaded from the dataset.

# Hyperparameters
seq_length = 24
batch_size = 32
forecast_horizon = 1
forecast_time = 5
epochs = 30

# Dataset configuration: loading.
loc_tz = pytz.timezone('Europe/Zurich')
split_date = loc_tz.localize(dt.datetime(2017, 1, 1, 0, 0, 0, 0))
path = os.path.join(os.path.abspath('.'), 'datasets/newdata.csv')  # Directory for dataset

print("Data Path:", path)
df = load_dataset(path=path, modules=features)
print("Datasize shape:", df.shape)
df_scaled = df.copy()
df_scaled = df_scaled.dropna()
print("Datasize shape (dropna):", df.shape)
print(df_scaled.head(1))

# Dataset configuration: standardizing all floats.
floats = [key for key in dict(df_scaled.dtypes) if dict(df_scaled.dtypes)[key] in ['float64']]  
scaler = StandardScaler()  # Min-Max Scaler
scaled_columns = scaler.fit_transform(df_scaled[floats])
df_scaled[floats] = scaled_columns
print("df_scaled.shape", df_scaled.shape)
print("Scaler mean =", scaler.mean_)
print("Scaler var =", scaler.var_)

# Dataset configuration: splitting training and testing dataset.
# df_train = df_scaled.loc[:5000].copy()
# df_test = df_scaled.loc[df_scaled.index >= split_date].copy()
df_train = df_scaled[:6000].copy()
df_test = df_scaled[6000:].copy()

X_train = df_train.drop('actual', 1).copy()#calender, weather
y_train = df_train['actual'].copy()
X_test = df_test.drop('actual', 1).copy()
y_test = df_test['actual'].copy()


print("X_train head:", X_train.head(1))

X_train=np.array(X_train, dtype=float)
y_train=np.array(y_train, dtype=float)
X_test=np.array(X_test, dtype=float)
y_test=np.array(y_test, dtype=float)
print("Training data shape:", np.shape(X_train))
print("Training label shape:", np.shape(y_train))
print("Training data shape:", np.shape(X_test))
print("Training label shape:", np.shape(y_test))

x_train, Y_train = reorganize2(X_train, y_train, seq_length, forecast_horizon, forecast_time)
#fore_horizon: size of output 
#fore_time: predict several days later
print("reorganize2"*10)
x_test, y_test = reorganize2(X_test, y_test, seq_length, forecast_horizon, forecast_time)
x_train = np.array(x_train, dtype=float) #include 'actual'
y_train = np.array(Y_train, dtype=float).reshape(-1, forecast_horizon)
x_test = np.array(x_test, dtype=float)
y_test = np.array(y_test, dtype=float).reshape(-1, forecast_horizon)
feature_dim = x_train.shape[2]#numofsample*seqlength*num_features
print("*"*30)
print(feature_dim)

print("Training data shape:", np.shape(x_train))
print("Training label shape:", np.shape(y_train))
print("Training data shape:", np.shape(x_test))
print("Training label shape:", np.shape(y_test))
# Model configuration.
sess = tf.Session()
keras.backend.set_session(sess)

x = tf.placeholder(tf.float32, shape=(None, seq_length, feature_dim))
y = tf.placeholder(tf.float32, shape=(None, forecast_horizon))
target = tf.placeholder(tf.float32, shape=(None, forecast_horizon))

model = rnn_model(seq_length=seq_length, input_dim=feature_dim, output_dim=forecast_horizon)
predictions = model(x)
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='adam')


# ==============================
# Training
# ==============================
model.load_weights('rnn_cleanfivesteps.h5')
opt_length = len(x_test)

y_pred = model.predict(x_test[:opt_length-seq_length], batch_size=32)
y_orig = y_test[:opt_length-seq_length]
x_temp = x_test[0:len(y_pred), 1, 1:].reshape(-1, 4)#?


print("y_pred shape =", np.shape(y_pred))
print("x_temp shape =", np.shape(x_temp))
x_pred = np.concatenate((y_pred, x_temp), axis=1)
x_orig = np.concatenate((y_orig, x_temp), axis=1)


df_1 = pd.DataFrame(x_pred, columns=df.columns.values)
pred_data = scaler.inverse_transform(df_1[floats])

df_3 = pd.DataFrame(x_orig, columns=df.columns.values)
original_data = scaler.inverse_transform(df_3[floats])


pred_data = np.array(pred_data, dtype=float)
original_data = np.array(original_data, dtype=float)

mae_val = calculate_mae(pred_data[:, 0], original_data[:, 0])
print("Prediction MAPE is: %f" % (mae_val))
with open('results/pred_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(pred_data)


with open('results/original_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(original_data)
y_orig = y_test[seq_length:opt_length]
x_temp = x_test[seq_length:opt_length, 1, 1:].reshape(-1, 4)#?
x_orig = np.concatenate((y_orig, x_temp), axis=1)

print("x_orig shape=", np.shape(x_orig))
x_orig = np.concatenate((y_orig, x_temp), axis=1)

df_3 = pd.DataFrame(x_orig, columns=df.columns.values)
original_data = scaler.inverse_transform(df_3[floats])
original_data = np.array(original_data, dtype=float)
mae_val = calculate_mae(pred_data[:, 0], original_data[:, 0])
print("Another prediction MAPE is: %f" % (mae_val))

