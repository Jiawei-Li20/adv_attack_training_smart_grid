
import pandas
import scipy
from scipy.io import loadmat
import math

import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
import numpy as np
b = np.genfromtxt('results/pred_data.csv',dtype=float,delimiter=',')
print(b.shape)
start=6028
length=2724
end=start+length
b= b[:,0].reshape(-1)
print(b.shape)
zero=np.zeros(start).reshape(-1)
print(zero.shape)
b=np.append(zero,b)
print(b.shape)
print(end)


# b[0]=0.09
# np.save('newsolar',b)
# np.save('changedsolar',16000*b)

#plt.plot(b,color='red')
# solar=np.load("solar.npy")
# plt.plot(solar)
# plt.show()
# print(solar.mean())
# print(b.mean())

# m=loadmat("QuantileSolutionRampStatic161028.mat")

# wind=np.array(m['wexp']).reshape(-1)
# print(wind.mean())
# print(solar.max())
# print(b.max())

# print(wind.max())
# plt.plot(emi_array_no,color='red',alpha=0.4)
# emi_array_battery=np.load("8wind_emi_array_battery.npy")
# emi_array_battery_rho=np.load("8wind_emi_array_battery_rho.npy")
# emi_array_no=np.load("8wind_emi_array_no.npy")

# plt.hist(emi_array_battery,bins=40,edgecolor="black")
# plt.show()
# plt.hist(emi_array_battery_rho,color='yellow',bins=40)
# plt.show()
# plt.hist(emi_array_no,color='red',bins=40)
# plt.show()
# plt.plot(emi_array_battery,alpha=0.5)

# plt.plot(emi_array_battery_rho,color='yellow',alpha=0.7)

# plt.plot(emi_array_no,color='red',alpha=0.4)

# plt.legend(["battery_without_rho","battery_with_rho","no_battery"])
# plt.xlabel('time')
# plt.ylabel('emission')
# plt.show(