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

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras.optimizers import SGD

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
epochs = 30
attack_times = 30

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
print(df_scaled['actual'[:100]])

# Dataset configuration: standardizing all floats.
floats = [key for key in dict(df_scaled.dtypes) if dict(df_scaled.dtypes)[key] in ['float64']]  
scaler = StandardScaler()  # Min-Max Scaler
scaled_columns = scaler.fit_transform(df_scaled[floats])
df_scaled[floats] = scaled_columns
print("df_scaled.shape", df_scaled.shape)
print("Scaler mean =", scaler.mean_)
print("Scaler var =", scaler.var_)
print(df_scaled['actual'[:100]])

# Dataset configuration: splitting training and testing dataset.
# df_train = df_scaled.loc[:5000].copy()
# df_test = df_scaled.loc[df_scaled.index >= split_date].copy()
df_train = df_scaled[:6000].copy()
df_test = df_scaled[6000:].copy()

X_train = df_train.drop('actual', 1).copy()#calender, weather
y_train = df_train['actual'].copy()
X_test = df_test.drop('actual', 1).copy()
y_test = df_test['actual'].copy()
print(y_test[:100])

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
# Training (Get the un-advtrained model)
# ==============================
print('\n' * 2 + '=' * 30 + '\nTraining begins. Get the un-advtrained model.\n' + '=' * 30)

# model.load_weights('rnn_cleanfivesteps.h5')
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
model.save_weights('rnn_cleanfivesteps.h5')
model.save_weights('rnn_substitute.h5')

y_result = model.predict(x_test)
y_real = y_test
plt.plot(y_result[:300, 0], 'r')
plt.plot(y_real[:300, 0], 'b')
plt.savefig('adv_results/training_result.png', dpi=150)
# plt.show()
print(y_test[:100])
print(y_result[:100])
raise ValueError

fig = plt.figure(figsize=(20.0, 6.4))


# mae_val = calculate_mae(y_result, y_real)
# print("Prediction MAPE is: %f, with noise %f" % (mae_val, 0))

# ==============================
# Get the Adversarial Training Examples
# ==============================
print('\n' * 2 + '=' * 30 + '\nGetting adversarial examples.\n' + '=' * 30)

x_train_orig = x_train.copy()
y_train_orig = y_train.copy()

# the attack noise would be outer_loop%
for outer_loop in range(1, 11):
    with sess.as_default():
        x_advtrain = []
        grad_new = []
        X_train2 = np.copy(x_train)
        print('*' * 10 + "[Current loop: # %d]" % (outer_loop) + '*' * 10)

        # Attack parameters
        eps = 0.01 * outer_loop  # Feature value change
        opt_length = len(x_train)
        bound = 0.01 * outer_loop
        # temp_bound_val = 0.5 * outer_loop

        model.load_weights('rnn_substitute.h5')

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


    # # test with adversarial samples
    # model.load_weights('rnn_cleanfivesteps.h5')
    # y_adv = model.predict(x_advtrain, batch_size=64)
    # y_pred = model.predict(x_test[:opt_length-seq_length], batch_size=32)
    # y_orig = y_test[:opt_length-seq_length]
    # x_temp = x_test[0:len(x_advtrain), 0, 1:].reshape(-1, 4)#?

    # x_temp_new = x_advtrain.copy()
    # for i in range(len(x_advtrain)-seq_length-1):
    #     for time_step in range(1, seq_length + 1):
    #         x_temp_new[i+time_step+1, -time_step, 0]=y_adv[i]#?

    # x_temp_new = x_temp_new[0:len(x_advtrain), 0, 1:].reshape(-1, 4)#?

    # print("y_pred shape =", np.shape(y_pred))
    # print("x_temp shape =", np.shape(x_temp))
    # x_pred = np.concatenate((y_pred, x_temp), axis=1)
    # x_adversarial = np.concatenate((y_adv, x_temp_new), axis=1)
    # x_orig = np.concatenate((y_orig, x_temp), axis=1)

    # df_1 = pd.DataFrame(x_pred, columns=df.columns.values)
    # pred_data = scaler.inverse_transform(df_1[floats])
    # df_2 = pd.DataFrame(x_adversarial, columns=df.columns.values)
    # adversarial_data = scaler.inverse_transform(df_2[floats])
    # df_3 = pd.DataFrame(x_orig, columns=df.columns.values)
    # original_data = scaler.inverse_transform(df_3[floats])
    # # print("x_adversarial shape:", x_adversarial.shape)
    # # print("adversarial_data shape:", adversarial_data.shape)

    # adversarial_data = np.array(adversarial_data, dtype=float)
    # pred_data = np.array(pred_data, dtype=float)
    # original_data = np.array(original_data, dtype=float)
    # import ipdb;ipdb.set_trace()
    # adv_mae_val = calculate_mae(adversarial_data[:, 0], original_data[:, 0])
    # print("for advexamples, Adversarial MAPE is: %f, with bound %f" % (adv_mae_val, eps))
    # mae_val = calculate_mae(pred_data[:, 0], original_data[:, 0])
    # print("for origdata, Prediction MAPE is: %f, with bound %f" % (mae_val, eps))

    # plt.clf()
    # plt.plot(adversarial_data[:, 0], 'r', label='Adversarial')
    # plt.plot(pred_data[:, 0], 'g', label='Predicted')
    # plt.plot(original_data[:, 0],'b', label='Original')
    # plt.ylabel('Load (MW)')
    # plt.legend()
    # plt.savefig('results_2022/load_attack_bound%.2f_mae%.4f_result.png' % (eps, adv_mae_val), dpi=150)
    # # plt.show()

    # with open('results_2022/load_attack_bound%.2f_result.csv' % (eps), 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(adversarial_data)
    # with open('results_2022/pred_data.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(pred_data)
    # with open('results_2022/original_data.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(original_data)
    # # adversarial_data = temp_bound(adversarial_data, original_data, temp_bound_val)
    



    x_newadvtrain = np.concatenate((x_train_orig, x_advtrain), axis=0)
    y_newadvtrain = np.concatenate((y_train_orig, y_train[:(opt_length - seq_length),:].reshape(-1,1)), axis=0)

    # ==============================
    # Adversarial training the model
    # ==============================

    # Re-initalize the model
    model_advtrain = rnn_model(seq_length=seq_length, input_dim=feature_dim, output_dim=forecast_horizon)
    predictions = model(x)
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model_advtrain.compile(loss='mean_squared_error', optimizer='adam')

    # first use the simple combination of two training sets
    # x_newadvtrain, y_newadvtrain copmuted in the previoius part

    model_advtrain.fit(x=x_newadvtrain, y=y_newadvtrain, batch_size=batch_size, epochs=epochs, shuffle=True)
    model_advtrain.save_weights(f'rnn_advtrained_eps_{outer_loop}.h5')

    # ==============================
    # Test
    # ==============================
    print('\n' * 2 + '=' * 30 + f'\nTest begins for eps {outer_loop}.\n' + '=' * 30)
    with sess.as_default():
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

        model.load_weights('rnn_advtrained.h5')

        counter = 0
        # Initialize the SGD optimizer
        grad, sign_grad = scaled_gradient(x, predictions, target)
        for q in range(opt_length - seq_length):
            if counter % 100 == 0 and counter > 0:
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

        X_new = np.array(X_new, dtype=float)  # <X_new> stores adversarial data.
        print("Adversarial X shape =", np.shape(X_new))
        #use the previous advtrain model?
        # model_advtrain.load_weights(f'rnn_advtrained_eps_{outer_loop}.h5')
        y_adv = model_advtrain.predict(X_new, batch_size=64)
        y_pred = model_advtrain.predict(x_test[:opt_length-seq_length], batch_size=32)
        y_orig = y_test[:opt_length-seq_length]
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
        plt.savefig('adv_results/load_attack_bound%.2f_result.png' % eps, dpi=150)
        # plt.show()

        with open('adv_results/load_attack_bound%.2f_result.csv' % (eps), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(adversarial_data)
        with open('adv_results/pred_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(pred_data)
        with open('adv_results/original_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(original_data)
        
        plt.clf()
        plt.plot(adversarial_data[:, 1], 'r', label="Adversarial")
        plt.plot(pred_data[:, 1], 'g', label="Predicted")
        plt.plot(original_data[:, 1], 'b', label="Original")
        plt.legend()
        plt.ylabel('Temperature (F)')
        plt.savefig('adv_results/load_attack_to_advtrain_noise%.2f_temp.png' % eps, dpi=150)


with open('adv_results/load_original.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(original_data)