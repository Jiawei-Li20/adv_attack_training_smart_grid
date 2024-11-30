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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras.optimizers import SGD

from datasets.data import *
from util import reorganize, reorganize2, calculate_mae
from nn_model import rnn_model, nn_model, svm_model, rnn_piecewise_model


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

def test_model(model, model_test, x, target, x_test, y_test, df, floats, model_path, outer_loop,data_file,scaler,epochs,attack_times,seq_length,feature_dim):
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
        # import ipdb;ipdb.set_trace()
        if outer_loop == 0:
            X_new = x_test[:opt_length - seq_length]
        else:
            for q in range(opt_length - seq_length):
                if counter % 1000 == 0 and counter > 0:
                    print("Optimization steps # %d ..." % (counter))
                #random_num=np.random.randint(2)
                random_num = 0

                Y_target = y_test[counter].reshape(-1, 1)

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
                

                if len(X_new) == 0:
                    X_new = X_new_group[0].reshape([1, seq_length, feature_dim])
                else:
                    X_new = np.concatenate((X_new, X_new_group[0].reshape([1, seq_length, feature_dim])), axis=0)

                counter += 1
        # print(x_test[:2])
        X_new = np.array(X_new, dtype=float)  # <X_new> stores adversarial data.
        # print("Adversarial X shape =", np.shape(X_new))
        #use the previous advtrain model?
        model_test.load_weights(data_file + model_path)
        y_tested = model_test.predict(X_new, batch_size=32)
        # y_pred = model_test.predict(x_test[:opt_length-seq_length], batch_size=32)
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
                X_new[i+time_step+1, -time_step, 0]=y_tested[i]#?

        x_temp_new = X_new[0:len(X_new), 0, 1:].reshape(-1, 4)#?

        # print("y_pred shape =", np.shape(y_pred))
        # print("x_temp shape =", np.shape(x_temp))
        x_tested = np.concatenate((y_tested, x_temp_new), axis = 1)
        # x_pred = np.concatenate((y_pred, x_temp), axis=1)
        # x_adversarial = np.concatenate((y_adv, x_temp_new), axis=1)
        # x_orig = np.concatenate((y_orig, x_temp), axis=1)
        df_0 = pd.DataFrame(x_tested, columns=df.columns.values)
        tested_data = scaler.inverse_transform(df_0[floats])

        # df_1 = pd.DataFrame(x_pred, columns=df.columns.values)
        # pred_data = scaler.inverse_transform(df_1[floats])
        # df_2 = pd.DataFrame(x_adversarial, columns=df.columns.values)
        # adversarial_data = scaler.inverse_transform(df_2[floats])
        # df_3 = pd.DataFrame(x_orig, columns=df.columns.values)
        # original_data = scaler.inverse_transform(df_3[floats])

        # print("x_adversarial shape:", x_adversarial.shape)
        # print("adversarial_data shape:", adversarial_data.shape)

        # adversarial_data = np.array(adversarial_data, dtype=float)
        # pred_data = np.array(pred_data, dtype=float)
        # original_data = np.array(original_data, dtype=float)
        tested_data = np.array(tested_data, dtype = float)


        # mae_val = calculate_mae(adversarial_data[:, 0], original_data[:, 0])
        # print("for advtrain model, Adversarial MAPE is: %f, with bound %f" % (mae_val, eps))
        # mae_val = calculate_mae(pred_data[:, 0], original_data[:, 0])
        # print("for advtrain model, Prediction MAPE is: %f, with bound %f" % (mae_val, eps))

        # plt.clf()
        # plt.plot(tested_data[:, 0], 'r', label='Adversarial')
        # # plt.plot(pred_data[:, 0], 'g', label='Predicted')
        # plt.plot(original_data[:, 0],'b', label='Original')
        # plt.ylabel('Load (MW)')
        # plt.legend()
        # plt.savefig(data_file + '/load_attack_bound_wo_advtrain%.2f_result.png' % eps, dpi=150)
        # # plt.show()
        with open(data_file + '/load_attack_bound_wo_advtrain%.2f_result.csv' % (eps), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(tested_data)
        return tested_data[:,0]
        # with open(data_file + '/test_with_plain_model_load_attack_bound%.2f_result.csv' % (eps), 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(adversarial_data)
        # with open(data_file + '/test_with_plain_model_pred_data%.2f.csv' % (eps), 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(pred_data)
        # with open(data_file + '/test_with_plain_model_original_data%.2f.csv' % (eps), 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(original_data)
        
        # plt.clf()
        # plt.plot(adversarial_data[:, 1], 'r', label="Adversarial")
        # plt.plot(pred_data[:, 1], 'g', label="Predicted")
        # plt.plot(original_data[:, 1], 'b', label="Original")
        # plt.legend()
        # plt.ylabel('Temperature (F)')
        # plt.savefig(data_file + '/load_attack_bound_wo_advtrain%.2f_temp.png' % eps, dpi=150)
        # raise ValueError


# def test_pred_model(model, model_test, x, target, x_test, y_test, df, floats, model_path, outer_loop,data_file,scaler,epochs,attack_times,seq_length,feature_dim):
#     print('\n' * 2 + '=' * 30 + f'\nTest begins for eps {outer_loop}.\n' + '=' * 30)
#     with tf.Session() as sess:
#         # the ratio of outer_loop% attack would be added.
#         X_new = []
#         grad_new = []
#         X_train2 = np.copy(x_test)

#         print('*' * 10 + "[Current loop: # %d]" % (outer_loop) + '*' * 10)
        
#         # Attack parameters
#         eps = 0.01 * outer_loop  # Feature value change
#         opt_length = len(x_test)
#         bound = 0.01 * outer_loop
#         # temp_bound_val = 0.5 * outer_loop

#         model.load_weights(data_file + 'rnn_cleanfivesteps.h5')
#         predictions = model(x)

#         counter = 0
#         # Initialize the SGD optimizer
#         grad, sign_grad = scaled_gradient(x, predictions, target)
#         if outer_loop == 0:
#             X_new = x_test
#         else:
#             for q in range(opt_length - seq_length):
#                 if counter % 1000 == 0 and counter > 0:
#                     print("Optimization steps # %d ..." % (counter))
#                 #random_num=np.random.randint(2)
#                 random_num = 0

#                 Y_target = y_test[counter].reshape(-1, 1)

#                 # Define input: x_t, x_{t+1},...,x_{t+pred_scope}.
#                 X_input = X_train2[counter]
#                 X_input = X_input.reshape(1, seq_length, feature_dim)
#                 X_new_group = np.copy(X_input) + get_Linf_rand(X_input.shape, bound = eps)

                
#                 # Outer iteration <it> for # gradient steps (data coming from API).
#                 # Inner iterations <j> for each dimension of the data
#                 for it in range(attack_times):
#                     gradient_value, grad_sign = sess.run([grad, sign_grad],
#                                                             feed_dict={x: X_new_group,
#                                                                         target: Y_target,
#                                                                         keras.backend.learning_phase(): 0})
#                     signed_grad = np.zeros(np.shape(X_input))
#                     signed_grad[:, :, 0] = outer_loop*40*grad_sign[:, :, 0]
#                     signed_grad[:, :, 1] = 0.15*grad_sign[:, :, 1]

#                     # gradient = np.zeros(np.shape(X_input))
#                     # gradient[:, :, 0] = gradient_value[:, :, 0]
#                     # signed_grad[:, :, 1] = grad_sign[:, :, 1]

#                     X_new_group = X_new_group + signed_grad
#                     X_new_group = check_constraint(X_input, X_new_group, bound)
#                         # else:#?
#                         #     X_new_group = X_new_group - eps * signed_grad
                

#                 if len(X_new) == 0:
#                     X_new = X_new_group[0].reshape([1, seq_length, feature_dim])
#                 else:
#                     X_new = np.concatenate((X_new, X_new_group[0].reshape([1, seq_length, feature_dim])), axis=0)

#                 counter += 1
#         # print(x_test[:2])
#         X_new = np.array(X_new, dtype=float)  # <X_new> stores adversarial data.
#         # print("Adversarial X shape =", np.shape(X_new))
#         #use the previous advtrain model?
#         model_test.load_weights(data_file + model_path)
#         y_tested = model_test.predict(X_new, batch_size=32)
#         # y_pred = model_test.predict(x_test[:opt_length-seq_length], batch_size=32)
#         y_orig = y_test[:opt_length-seq_length]
#         # import ipdb;ipdb.set_trace()
#         # if adv_attack[:4] == "cost":
#         #     y_adv = inverse_cost_func(y_adv)
#         #     y_pred = inverse_cost_func(y_pred)
#         #     y_orig = inverse_cost_func(y_orig)
#         #     y_adv = (y_adv - mean) / np.sqrt(var)
#         #     y_pred = (y_pred - mean) / np.sqrt(var)
#         #     y_orig = (y_orig - mean) / np.sqrt(var)
#         x_temp = x_test[0:len(X_new), 0, 1:].reshape(-1, 4)#?

#         for i in range(len(X_new)-seq_length-1):
#             for time_step in range(1, seq_length + 1):
#                 X_new[i+time_step+1, -time_step, 0]=y_tested[i]#?

#         x_temp_new = X_new[0:len(X_new), 0, 1:].reshape(-1, 4)#?

#         # print("y_pred shape =", np.shape(y_pred))
#         # print("x_temp shape =", np.shape(x_temp))
#         x_tested = np.concatenate((y_tested, x_temp_new), axis = 1)
#         # x_pred = np.concatenate((y_pred, x_temp), axis=1)
#         # x_adversarial = np.concatenate((y_adv, x_temp_new), axis=1)
#         # x_orig = np.concatenate((y_orig, x_temp), axis=1)
#         df_0 = pd.DataFrame(x_tested, columns=df.columns.values)
#         tested_data = scaler.inverse_transform(df_0[floats])

#         # df_1 = pd.DataFrame(x_pred, columns=df.columns.values)
#         # pred_data = scaler.inverse_transform(df_1[floats])
#         # df_2 = pd.DataFrame(x_adversarial, columns=df.columns.values)
#         # adversarial_data = scaler.inverse_transform(df_2[floats])
#         # df_3 = pd.DataFrame(x_orig, columns=df.columns.values)
#         # original_data = scaler.inverse_transform(df_3[floats])

#         # print("x_adversarial shape:", x_adversarial.shape)
#         # print("adversarial_data shape:", adversarial_data.shape)

#         # adversarial_data = np.array(adversarial_data, dtype=float)
#         # pred_data = np.array(pred_data, dtype=float)
#         # original_data = np.array(original_data, dtype=float)
#         tested_data = np.array(tested_data, dtype = float)
#         return tested_data[:,0]
