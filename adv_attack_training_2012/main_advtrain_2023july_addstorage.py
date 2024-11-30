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
from scipy.io import loadmat
from statsmodels.tsa import stattools

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
# tf.enable_eager_execution()
tf.set_random_seed(11)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras.optimizers import SGD

from datasets.data import *
from util import reorganize, reorganize2, calculate_mae
from nn_model import rnn_model, nn_model, svm_model, rnn_piecewise_model
from hyperparameters import loadratio, windratio, solar_ratio

# prepare cost
m=loadmat("QuantileSolutionRampStatic161028.mat")
capacity=np.array(m['capacity']).reshape(-1)
capacity=np.rint(capacity).astype(np.int64)
mc=np.array(m['mc']).reshape(-1)
id_=np.argsort(mc)
capacity=capacity[id_]
mc = mc[id_]
tempcost = 0
plant_cost_curve=np.array([0])
marginal_cost = np.array([0])
bound_capacity = [0]
# base = [0]
# mean = 20870.699801846433
# var = 302071118.24226564
for i in range(0,len(mc)):
    for j in range(0,capacity[i]):
        tempcost=tempcost+mc[i]
        plant_cost_curve=np.append(plant_cost_curve,tempcost)
        marginal_cost=np.append(marginal_cost,mc[i])
        #print(plant_emi_curve)
    bound_capacity.append(bound_capacity[-1] + capacity[i])
    # base.append(base[-1] + (capacity[i]-mean)/np.sqrt(var) * mc[i])
print(plant_cost_curve[-10:])
# raise ValueError
# bound_capacity = (np.array(bound_capacity) - mean)/np.sqrt(var)
# base = np.array(base)
# plant_cost_curve = np.array(plant_cost_curve)

# def inverse_cost_func(y):
#     cost_y = y * dec_magnitude
#     product_y = np.zeros(y.shape)
#     for k in range(len(y)):
#         lb = 0
#         ub = 0
#         offset = 0
#         base = 0
#         for i in range(len(mc)):
#             if base + mc[i]*capacity[i] > cost_y[k,0]:
#                 product_y[k,0] = bound_capacity[i] + (cost_y[k,0]- base)/mc[i]
#                 base += mc[i]* capacity[i]
#                 break
#             base += mc[i]* capacity[i]
#         # import ipdb;ipdb.set_trace()
#         if base <= cost_y[k,0]:
#             product_y[k,0] = bound_capacity[-1] + (cost_y[k,0]- base)/mc[-1]
#     return product_y
''' test session
mean = 4.16241129e+04
var = 1.60763191e+08
ol = np.array([len(plant_cost_curve)-1])
l = (ol - mean)/np.sqrt(var)
l = np.array([0, 1,-0.10700128, -0.13569216, -0.07747312])#  0.18627576  0.44919885  0.47272507 0.4742568   0.39890327  0.26659     0.12225029]
ol = (l*np.sqrt(var)+ mean).astype(np.int64)
print(l)
test = tf.convert_to_tensor(l, dtype=tf.float64)
with tf.Session() as sess:
    # print(cost_func(test))
    cost = cost_func(test).eval()
    print(cost * dec_magnitude)
    print(plant_cost_curve[ol] + (plant_cost_curve[ol+1] - plant_cost_curve[ol]).dot(l*np.sqrt(var)+ mean-ol))
    # print(plant_cost_curve[ol+1])

    # import ipdb;ipdb.set_trace()
    # print(inverse_cost_func(cost.reshape(-1,1)))
    # print("std product",ol)
    # print("std normalized product", l)
raise ValueError
'''

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

# to generate Linf rand within bound [-bound,bound)
def get_Linf_rand(shape, bound):
    return  2 * bound * np.random.random(shape) - bound * np.ones(shape)
# def bound_val(x_orig, bound):
#     x_low=[]
#     x_high=[]
#     for i in range(np.shape(x_orig)[0]):
#         x_low.append(x_orig[i,0,0]-bound*np.abs(x_orig[i, 0, 0]))
#         x_high.append(x_orig[i, 0, 0] + bound * np.abs(x_orig[i, 0, 0]))
#     x_low=np.array(x_low, dtype=float)
#     x_high=np.array(x_high, dtype=float)

#     return x_low, x_high

# def temp_bound(adv, orig, temp_val):
#     for i in range(1, 2):
#         for j in range(len(adv)):
#             adv[j, i]=np.clip(adv[j,i], orig[j,i]-temp_val, orig[j,i]+temp_val)
#     return adv

# ==============================
# Main Settings and Initialization
# ==============================
print('=' * 30 + '\nInitialization begins.\n' + '=' * 30)
features = ['actual', 'calendar', 'weather']  # features to be loaded from the dataset.

# Hyperparameters
seq_length = 24
batch_size = 32
forecast_horizon = 1
forecast_time = 1
epochs = 80
attack_times = 30
import argparse

parser = argparse.ArgumentParser(description='aims')
parser.add_argument('-trainaim', type=str, help='cost_strg or cost_strg_strategy')
args = parser.parse_args()
adv_example = args.trainaim#"pred" or "cost"
adv_attack = args.trainaim
data_file = "adv_" + adv_example + "_attack_" + adv_attack + "_net"
if not os.path.exists(data_file):
    os.makedirs(data_file)
# Dataset configuration: loading.
loc_tz = pytz.timezone('Europe/Zurich')
split_date = loc_tz.localize(dt.datetime(2017, 1, 1, 0, 0, 0, 0))
path = os.path.join(os.path.abspath('.'), 'datasets/newdata.csv')  # Directory for dataset

print("Data Path:", path)
df = load_dataset(path=path, modules=features)
print("Datasize shape:", df.shape)
df_scaled = df.copy()
df_scaled = df_scaled.dropna()
print("Datasize shape (dropna):", df_scaled.shape)
only_na = df[~df.index.isin(df_scaled.index)]
print(only_na)
print(df_scaled.head(1))
df_scaled['actual'] *= loadratio



wind=np.array(m['wexp']).reshape(-1)
solar=np.load("changedsolar.npy")*solar_ratio#16000
green_energy=windratio*(wind+solar)
print(green_energy.shape, df_scaled.shape)
#the last 5 data only have actual load and others are NA,so they are dropped
df_scaled['actual'] -= green_energy[:df_scaled['actual'].shape[0]]

# raise ValueError
# print(df_scaled['actual'[:100]])
# # Dataset configuration: standardizing all floats.
floats = [key for key in dict(df_scaled.dtypes) if dict(df_scaled.dtypes)[key] in ['float64']]  
scaler = StandardScaler()  # Min-Max Scaler
scaled_columns = scaler.fit_transform(df_scaled[floats])
df_scaled[floats] = scaled_columns
print("df_scaled.shape", df_scaled.shape)
print("Scaler mean =", scaler.mean_)
print("Scaler var =", scaler.var_)
mean = scaler.mean_[0]
var = scaler.var_[0]
# print("df_scaled actual = ", df_scaled['actual'])
# import ipdb;ipdb.set_trace()
# print(df_scaled['actual'[:100]])
# raise ValueError
# Dataset configuration: splitting training and testing dataset.
barB=16000
marginal_cost = np.array(marginal_cost)
well_defined_length = marginal_cost.shape[0]
marginal_cost = tf.convert_to_tensor(marginal_cost)
# import ipdb;ipdb.set_trace()
def marginal_cost_fn(load):
    # x = np.rint(load)
    # if load<0:
    #     return 0
    # if load>=well_defined_length:
    #     return marginal_cost[-1]
    # return marginal_cost[tf.cast(tf.floor(load),tf.int32)]
    load = load * tf.sqrt(var) + mean*tf.ones_like(load)
    res = tf.zeros_like(load)

    for i in range(len(mc)):
        lb = bound_capacity[i]
        ub = bound_capacity[i+1]
        res += tf.where(tf.math.logical_and(tf.math.greater_equal(load,lb), tf.math.less(load, ub)), mc[i] * tf.ones_like(load), tf.zeros_like(load))
    res += tf.where(tf.math.greater_equal(load,bound_capacity[-1]), mc[-1] * tf.ones_like(load), tf.zeros_like(load))
    return res
dec_magnitude = 100000

def cost_func(y):
    res = tf.zeros_like(y)
    lb = 0
    ub = 0
    offset = 0
    base = 0
    # mean = 4.16241129e+04
    # var = 1.60763191e+08
    original_y = mean*tf.ones_like(y) + y * np.sqrt(var)
    # print(original_y.eval())
    for i in range(len(mc)):
        lb = bound_capacity[i]
        ub = bound_capacity[i+1]
        offset = lb
        #base = base_vec[i]
        # ub += capacity[i]
        res += tf.where(tf.math.logical_and(tf.math.greater_equal(original_y,lb), tf.math.less(original_y, ub)), base + mc[i] * (original_y-offset), tf.zeros_like(original_y))
        # offset += capacity[i]
        base += mc[i]*capacity[i]
        # lb = ub
    # print(res.eval())
    res += tf.where(tf.math.greater_equal(original_y,bound_capacity[-1]), base + mc[-1] * (original_y-bound_capacity[-1]), tf.zeros_like(original_y))
    # print(res.eval())
  # print(res)
    return res/dec_magnitude
    # return plant_cost_curve, maxcapcity

def custom_loss_wrapper(x):
    # def custom_loss(y_true,y_pred):
    #     # print(x.shape)#check batch_size * T * 5
    #     (_, T, _) = x.shape
    #     storage = tf.zeros_like(x[:,0,0], dtype = tf.float64)
    #     load = x * tf.sqrt(var) + mean*tf.ones_like(x)
    #     # i = 0
    #     for i in range(T):
    #         storage = tf.math.maximum(tf.zeros_like(load[:,0,0],dtype = tf.float64),tf.math.minimum(barB*tf.ones_like(load[:,0,0],dtype = tf.float64), storage - load[:,i,0]))
    #     storage = (storage - mean*tf.ones_like(storage))/tf.sqrt(var)

    #     return tf.reduce_mean(tf.square(cost_func(y_true-storage) - cost_func(y_pred-storage)), axis=-1)
    # return custom_loss
    def custom_loss(y_true,y_pred):
        # print(x.shape)#check batch_size * T * 5
        # import ipdb;ipdb.set_trace()
        (_, T, _) = x.shape

        # batchsize, _ = y_true
        storage = tf.zeros_like(x[:,0,0], dtype = tf.float64)
        load = x * tf.sqrt(var) + mean*tf.ones_like(x)
        # i = 0
        for i in range(T):
            storage = tf.math.maximum(tf.zeros_like(load[:,0,0],dtype = tf.float64),tf.math.minimum(barB*tf.ones_like(load[:,0,0],dtype = tf.float64), storage - load[:,i,0]))


        #storage = storage - x[:,0,0]
        # tmp1 = barB*tf.ones(batchsize ,dtype = tf.float64)
        # storage = tf.math.minimum(tmp1, storage)
        # storage = tf.math.maximum(tf.zeros(batchsize ,dtype = tf.float64),storage)
        
        # storage = tf.cast(storage, tf.float64)
        # print(y_pred.shape,y_true.shape)
        # import ipdb;ipdb.set_trace()
        norm_p = y_pred * tf.sqrt(var) + mean*tf.ones_like(y_pred)
        norm_t = y_true * tf.sqrt(var) + mean*tf.ones_like(y_true)
        exceed = tf.cast((norm_p-norm_t>barB- storage),dtype = tf.float64)
        lower = tf.cast((norm_p-norm_t<-storage), dtype = tf.float64)
        exceed_slope = marginal_cost_fn(y_pred)
        lower_slope = marginal_cost_fn((norm_t-storage-mean*tf.ones_like(storage))/tf.sqrt(var))

        return tf.reduce_mean(tf.square(cost_func(y_true) - cost_func(y_pred))+ 
        tf.square(exceed *(y_pred * tf.sqrt(var) + mean*tf.ones_like(y_pred)-norm_t-barB+storage)*exceed_slope/dec_magnitude
        +lower*(norm_t-(y_pred * tf.sqrt(var) + mean*tf.ones_like(y_pred))-storage)*lower_slope/dec_magnitude), axis=-1)
    return custom_loss

def custom_loss_wrapper_strategy(x):
    def custom_loss(y_true,y_pred):
        # print(x.shape)#check batch_size * T * 5
        # import ipdb;ipdb.set_trace()
        (_, T, _) = x.shape

        # batchsize, _ = y_true
        storage = tf.zeros_like(x[:,0,0], dtype = tf.float64)
        load = x * tf.sqrt(var) + mean*tf.ones_like(x)
        # i = 0
        for i in range(T):
            storage = tf.math.maximum(tf.zeros_like(load[:,0,0],dtype = tf.float64),tf.math.minimum(barB*tf.ones_like(load[:,0,0],dtype = tf.float64), storage - load[:,i,0]))


        #storage = storage - x[:,0,0]
        # tmp1 = barB*tf.ones(batchsize ,dtype = tf.float64)
        # storage = tf.math.minimum(tmp1, storage)
        # storage = tf.math.maximum(tf.zeros(batchsize ,dtype = tf.float64),storage)
        
        # storage = tf.cast(storage, tf.float64)
        # print(y_pred.shape,y_true.shape)
        # import ipdb;ipdb.set_trace()
        norm_p = y_pred * tf.sqrt(var) + mean*tf.ones_like(y_pred)
        norm_t = y_true * tf.sqrt(var) + mean*tf.ones_like(y_true)
        exceed = tf.cast((norm_p-norm_t>barB- storage),dtype = tf.float64)
        lower = tf.cast((norm_p-norm_t<-storage), dtype = tf.float64)
        exceed_slope = marginal_cost_fn(y_pred)
        lower_slope = marginal_cost_fn((norm_t-storage-mean*tf.ones_like(storage))/tf.sqrt(var))

        return tf.reduce_mean(tf.square(cost_func(y_true) - cost_func(y_pred)- 
        exceed *(y_pred * tf.sqrt(var) + mean*tf.ones_like(y_pred)-norm_t-barB+storage)*exceed_slope/dec_magnitude-lower*(norm_t-(y_pred * tf.sqrt(var) + mean*tf.ones_like(y_pred))-storage)*lower_slope/dec_magnitude), axis=-1)
    return custom_loss
# import ipdb;ipdb.set_trace()

# x = tf.constant([[[-mean/np.sqrt(var)]]],dtype=tf.float64)
# y2 = tf.constant([0.2],dtype=tf.float64)
# y1 = tf.constant([0.1],dtype=tf.float64)
# custom_loss_wrapper_strategy(x)(y1,y2)
# import ipdb;ipdb.set_trace()
def cost_loss(y_true,y_pred):
    #print(y_pred.shape)
    return tf.reduce_mean(tf.square(cost_func(y_true) - cost_func(y_pred)), axis=-1)
    #return tf.reduce_mean(tf.square(plant_cost_curve[y_true] - plant_cost_curve[y_pred]), axis=-1



df_train = df_scaled[:6000].copy()
df_test = df_scaled[6000:].copy()
# # for debug
# df_train = df_scaled[:100].copy()
# df_test = df_scaled[-100:].copy()

X_train = df_train.drop('actual', 1).copy()#calender, weather
y_train = df_train['actual'].copy()
X_test = df_test.drop('actual', 1).copy()
y_test = df_test['actual'].copy()

print("normalized y real", y_test[:10])
print("X_train head:", X_train.head(1))

X_train=np.array(X_train, dtype=float)
y_train=np.array(y_train, dtype=float)
X_test=np.array(X_test, dtype=float)
y_test=np.array(y_test, dtype=float)
print("Training data shape:", np.shape(X_train))
print("Training label shape:", np.shape(y_train))
print("Training data shape:", np.shape(X_test))
print("Training label shape:", np.shape(y_test))
# import ipdb;ipdb.set_trace()

# if adv_example == "cost":
#     with tf.Session() as sess:
#         y_train = np.array(cost_func(y_train).eval())
#         y_test = np.array(cost_func(y_test).eval())
#     print("cost", y_test[:10])
    # import ipdb;ipdb.set_trace()
    # print(y_test.to_numpy())
# raise ValueError
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

#x = tf.placeholder(tf.float64, shape=(None, seq_length, feature_dim))
#y = tf.placeholder(tf.float64, shape=(None, forecast_horizon))
target = tf.placeholder(tf.float64, shape=(None, forecast_horizon))

x = tf.keras.Input(shape=(seq_length, feature_dim),dtype="float64")
model_rnn = rnn_model(seq_length=seq_length, input_dim=feature_dim, output_dim=forecast_horizon)
predictions = model_rnn(x)
model = tf.keras.Model(x, predictions)
# import ipdb;ipdb.set_trace()
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
if adv_example == "cost":
    model.compile(loss=cost_loss, optimizer='adam')
elif adv_example =="cost_strg":
    model.compile(loss=custom_loss_wrapper(x), optimizer='adam')
elif adv_example =="cost_strg_strategy":
    model.compile(loss=custom_loss_wrapper_strategy(x), optimizer='adam')
elif adv_example == "pred":
    model.compile(loss='mean_squared_error', optimizer='adam')

    



# ==============================
# Training (Get the un-advtrained model)
# ==============================
print('\n' * 2 + '=' * 30 + '\nTraining begins. Get the un-advtrained model.\n' + '=' * 30)

x_train_double = x_train.astype(np.float64)
y_train_double = y_train.astype(np.float64)

print(x_train_double.shape)
print(x_train_double.dtype)
print(y_train_double.shape)
print(y_train_double.dtype)

print(seq_length)
print(feature_dim)
print(forecast_horizon)

# model.load_weights('rnn_cleanfivesteps.h5')
model.fit(x=x_train_double, y=y_train_double, batch_size=batch_size, epochs=epochs, shuffle=True)
model.save_weights(data_file + 'rnn_cleanfivesteps.h5')
model.save_weights(data_file + 'rnn_substitute.h5')

y_result = model.predict(x_test)
y_real = y_test
if adv_example[:4] == "cost":
    plt.plot(y_result[:, 0], 'r')
    plt.plot(y_real[:, 0], 'b')
    plt.savefig(data_file + '/training_result.png', dpi=150)
    plt.clf()
    with sess.as_default():
        test = tf.convert_to_tensor(y_result[:, 0], dtype=tf.float64)
        cost = cost_func(test).eval()
        print(cost[:10])
        plt.plot(cost, 'r')
        test = tf.convert_to_tensor(y_real[:, 0], dtype=tf.float64)
        cost = cost_func(test).eval()
        # import ipdb;ipdb.set_trace()
        print(cost[:10])
        plt.plot(cost, 'r')
        plt.plot(cost, 'b')
    plt.savefig(data_file + '/training_result_cost.png', dpi=150)
else:
    plt.plot(y_result[:, 0], 'r')
    plt.plot(y_real[:, 0], 'b')
    plt.savefig(data_file + '/training_result.png', dpi=150)
mae_val = calculate_mae(y_result, y_real)
print("Prediction MAPE is: %f, with noise %f" % (mae_val, 0))

from test import test_model
mae = np.zeros(11)
start=6024
length=2728
end=start+length
for outer_loop in range(0,11):
    model_path = 'rnn_substitute.h5'
    model_test = tf.keras.Model(x, predictions)
    attacked_data = test_model(model, model_test, x, target, x_test, y_test, df, floats, model_path, outer_loop,data_file,scaler,epochs,attack_times, seq_length, feature_dim)
    pred = attacked_data + green_energy[start:end]
    # import ipdb;ipdb.set_trace()
    mae[outer_loop] = calculate_mae(pred, df['actual'][start:end]*loadratio)
    print(mae[outer_loop])
print(mae)
np.save(data_file+"/attacked_mae", mae)
# raise ValueError
# plt.show()
# print(y_result[:10,0])
# print(y_real[:10,0])
# a = (y_result[:10,0]*np.sqrt(var)+mean).astype(np.int64)
# print(plant_cost_curve[a])
# a = (y_real[:10,0]*np.sqrt(var)+mean).astype(np.int64)
# print(plant_cost_curve[a])

# print(cost_func(y_result[:10,0]))
# print(cost_func(y_real[:10,0]))

# raise ValueError


# ==============================
# Get the Adversarial Training Examples
# ==============================
print('\n' * 2 + '=' * 30 + '\nGetting adversarial examples.\n' + '=' * 30)

x_train_orig = x_train_double.copy()
y_train_orig = y_train_double.copy()

# the attack noise would be outer_loop%
for outer_loop in range(1, 11):
    with tf.Session() as sess:
        x_advtrain = []
        grad_new = []
        X_train2 = np.copy(x_train)
        print('*' * 10 + "[Current loop: # %d]" % (outer_loop) + '*' * 10)

        # Attack parameters
        eps = 0.01 * outer_loop  # Feature value change
        opt_length = len(x_train)
        bound = 0.01 * outer_loop
        # temp_bound_val = 0.5 * outer_loop

        model.load_weights(data_file+'rnn_substitute.h5')

        counter = 0
        # Initialize the SGD optimizer
        grad, sign_grad = scaled_gradient(x, predictions, target)
        for q in range(opt_length - seq_length):
            if counter % 1000 == 0 and counter > 0:
                print("Optimization steps # %d ..." % (counter))
            #random_num=np.random.randint(2)
            random_num = 0

            Y_target = y_train[counter].reshape(-1, 1)

            # Define input: x_t, x_{t+1},...,x_{t+pred_scope}.
            X_input = X_train2[counter]
            X_input = X_input.reshape(1, seq_length, feature_dim)
            X_new_group = np.copy(X_input) + get_Linf_rand(X_input.shape, bound = eps)

            # Outer iteration <it> for # gradient steps (data coming from API).
            # Inner iterations <j> for each dimension of the data
            for it in range(attack_times):
                gradient_value, grad_sign = sess.run([grad, sign_grad],
                                                        feed_dict={x: X_new_group,
                                                                target: Y_target,
                                                                keras.backend.learning_phase(): 0})
                signed_grad = np.zeros(np.shape(X_input))
                signed_grad[:, :, 0] = outer_loop*40*grad_sign[:, :, 0]
                signed_grad[:, :, 1] = 0.15*grad_sign[:, :, 1]
                # gradient = np.zeros(np.shape(X_input))
                # gradient[:, :, 0] = gradient_value[:, :, 0]
                # signed_grad[:, :, 1] = grad_sign[:, :, 1]
                X_new_group = X_new_group + signed_grad
                X_new_group = check_constraint(X_input, X_new_group, bound)
                    # else:#?
                    #     X_new_group = X_new_group - eps * signed_grad            

            if len(x_advtrain) == 0:
                x_advtrain = X_new_group[0].reshape([1, seq_length, feature_dim])
            else:
                x_advtrain = np.concatenate((x_advtrain, X_new_group[0].reshape([1, seq_length, feature_dim])), axis=0)

            counter += 1

    x_advtrain = np.array(x_advtrain, dtype=float)  # <X_new> stores adversarial data.

    # if adv_example == "pred" and adv_attack == "cost":
    #     with sess.as_default():
    #         y_train_orig = np.array(cost_func(y_train_orig).eval())
    #         y_test = np.array(cost_func(y_test).eval())
    #         y_train = np.array(cost_func(y_train).eval())

    # import ipdb;ipdb.set_trace()
    # if adv_example == "cost" and adv_attack == "pred":
    #     y_train = inverse_cost_func(y_train)
    #     y_test = inverse_cost_func(y_test)
    #     y_train_orig = inverse_cost_func(y_train_orig)
    #     y_train = (y_train - mean) / np.sqrt(var)
    #     y_test = (y_test - mean) / np.sqrt(var)
    #     y_train_orig = (y_train_orig - mean) / np.sqrt(var)

    x_newadvtrain = np.concatenate((x_train_orig, x_advtrain), axis=0)
    y_newadvtrain = np.concatenate((y_train_orig, y_train[:(opt_length - seq_length),:].reshape(-1,1)), axis=0)

    # ==============================
    # Adversarial training the model
    # ==============================

    # Re-initalize the model
    x = tf.keras.Input(shape=(seq_length, feature_dim),dtype="float64")
    model_advtrain_rnn = rnn_model(seq_length=seq_length, input_dim=feature_dim, output_dim=forecast_horizon)
    advtrain_predictions = model_advtrain_rnn(x)
    model_advtrain = tf.keras.Model(x,advtrain_predictions)
    
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    if adv_attack == "cost":
        model_advtrain.compile(loss=cost_loss, optimizer='adam')
    if adv_attack =="cost_strg":
        model_advtrain.compile(loss=custom_loss_wrapper(x), optimizer='adam')
    if adv_attack =="cost_strg_strategy":
        model_advtrain.compile(loss=custom_loss_wrapper_strategy(x), optimizer='adam')
    if adv_attack == "pred":
        model_advtrain.compile(loss='mean_squared_error', optimizer='adam')

    # first use the simple combination of two training sets
    # x_newadvtrain, y_newadvtrain copmuted in the previoius part

    model_advtrain.fit(x=x_newadvtrain, y=y_newadvtrain, batch_size=batch_size, epochs=epochs, shuffle=True)
    model_advtrain.save_weights(data_file + f'rnn_advtrained_eps_{outer_loop}.h5')

    # ==============================
    # Test
    # ==============================
    print('\n' * 2 + '=' * 30 + f'\nTest begins for eps {outer_loop}.\n' + '=' * 30)
    with tf.Session() as sess:
        # the ratio of outer_loop% attack would be added.
        X_new = []
        grad_new = []
        X_train2 = np.copy(x_test)
        print('*' * 10 + "[Current loop: # %d]" % (outer_loop) + '*' * 10)
        
        # Attack parameters
        eps = 0.01 * outer_loop  # Feature value change
        opt_length = len(x_test)
        bound = 0.01 * outer_loop
        # temp_bound_val = 0.5 * outer_loop

        model.load_weights(data_file + 'rnn_cleanfivesteps.h5')
        predictions = model(x)

        counter = 0
        # Initialize the SGD optimizer
        grad, sign_grad = scaled_gradient(x, predictions, target)
        for q in range(opt_length - seq_length):
            if counter % 1000 == 0 and counter > 0:
                print("Optimization steps # %d ..." % (counter))
            #random_num=np.random.randint(2)
            random_num = 0

            Y_target = y_test[counter].reshape(-1, 1)

            # Define input: x_t, x_{t+1},...,x_{t+pred_scope}.
            X_input = X_train2[counter]
            X_input = X_input.reshape(1, seq_length, feature_dim)
            X_new_group = np.copy(X_input) + get_Linf_rand(X_new_group.shape, bound = eps)

            
            # Outer iteration <it> for # gradient steps (data coming from API).
            # Inner iterations <j> for each dimension of the data
            for it in range(attack_times):
                gradient_value, grad_sign = sess.run([grad, sign_grad],
                                                        feed_dict={x: X_new_group,
                                                                    target: Y_target,
                                                                    keras.backend.learning_phase(): 0})
                signed_grad = np.zeros(np.shape(X_input))
                signed_grad[:, :, 0] = outer_loop*40*grad_sign[:, :, 0]
                signed_grad[:, :, 1] = 0.15*grad_sign[:, :, 1]

                # gradient = np.zeros(np.shape(X_input))
                # gradient[:, :, 0] = gradient_value[:, :, 0]
                # signed_grad[:, :, 1] = grad_sign[:, :, 1]

                X_new_group = X_new_group + signed_grad
                X_new_group = check_constraint(X_input, X_new_group, bound)
                    # else:#?
                    #     X_new_group = X_new_group - eps * signed_grad
            

            if len(X_new) == 0:
                X_new = X_new_group[0].reshape([1, seq_length, feature_dim])
            else:
                X_new = np.concatenate((X_new, X_new_group[0].reshape([1, seq_length, feature_dim])), axis=0)

            counter += 1
        print(x_test[:2])
        X_new = np.array(X_new, dtype=float)  # <X_new> stores adversarial data.
        print("Adversarial X shape =", np.shape(X_new))
        #use the previous advtrain model?
        model_advtrain.load_weights(data_file + f'rnn_advtrained_eps_{outer_loop}.h5')
        y_adv = model_advtrain.predict(X_new, batch_size=32)
        y_pred = model_advtrain.predict(x_test[:opt_length-seq_length], batch_size=32)
        y_orig = y_test[:opt_length-seq_length]
        # import ipdb;ipdb.set_trace()
        # if adv_attack[:4] == "cost":
        #     y_adv = inverse_cost_func(y_adv)
        #     y_pred = inverse_cost_func(y_pred)
        #     y_orig = inverse_cost_func(y_orig)
        #     y_adv = (y_adv - mean) / np.sqrt(var)
        #     y_pred = (y_pred - mean) / np.sqrt(var)
        #     y_orig = (y_orig - mean) / np.sqrt(var)
        x_temp = x_test[0:len(X_new), 0, 1:].reshape(-1, 4)#?

        for i in range(len(X_new)-seq_length-1):
            for time_step in range(1, seq_length + 1):
                X_new[i+time_step+1, -time_step, 0]=y_adv[i]#?

        x_temp_new = X_new[0:len(X_new), 0, 1:].reshape(-1, 4)#?

        print("y_pred shape =", np.shape(y_pred))
        print("x_temp shape =", np.shape(x_temp))
        x_pred = np.concatenate((y_pred, x_temp), axis=1)
        x_adversarial = np.concatenate((y_adv, x_temp_new), axis=1)
        x_orig = np.concatenate((y_orig, x_temp), axis=1)

        df_1 = pd.DataFrame(x_pred, columns=df.columns.values)
        pred_data = scaler.inverse_transform(df_1[floats])
        df_2 = pd.DataFrame(x_adversarial, columns=df.columns.values)
        adversarial_data = scaler.inverse_transform(df_2[floats])
        df_3 = pd.DataFrame(x_orig, columns=df.columns.values)
        original_data = scaler.inverse_transform(df_3[floats])
        # pred_data = x_pred
        # adversarial_data = x_adversarial
        # original_data = x_orig
        print("x_adversarial shape:", x_adversarial.shape)
        print("adversarial_data shape:", adversarial_data.shape)

        adversarial_data = np.array(adversarial_data, dtype=float)
        pred_data = np.array(pred_data, dtype=float)
        original_data = np.array(original_data, dtype=float)

        mae_val = calculate_mae(adversarial_data[:, 0], original_data[:, 0])
        print("for advtrain model, Adversarial MAPE is: %f, with bound %f" % (mae_val, eps))
        mae_val = calculate_mae(pred_data[:, 0], original_data[:, 0])
        print("for advtrain model, Prediction MAPE is: %f, with bound %f" % (mae_val, eps))

        plt.clf()
        plt.plot(adversarial_data[:, 0], 'r', label='Adversarial')
        plt.plot(pred_data[:, 0], 'g', label='Predicted')
        plt.plot(original_data[:, 0],'b', label='Original')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.savefig(data_file + '/test_with_plain_model_load_attack_bound%.2f_result.png' % eps, dpi=150)
        # plt.show()

        with open(data_file + '/test_with_plain_model_load_attack_bound%.2f_result.csv' % (eps), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(adversarial_data)
        with open(data_file + '/test_with_plain_model_pred_data%.2f.csv' % (eps), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(pred_data)
        with open(data_file + '/test_with_plain_model_original_data%.2f.csv' % (eps), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(original_data)
        
        plt.clf()
        plt.plot(adversarial_data[:, 1], 'r', label="Adversarial")
        plt.plot(pred_data[:, 1], 'g', label="Predicted")
        plt.plot(original_data[:, 1], 'b', label="Original")
        plt.legend()
        plt.ylabel('Temperature (F)')
        plt.savefig(data_file + '/test_with_plain_model_load_attack_to_advtrain_noise%.2f_temp.png' % eps, dpi=150)