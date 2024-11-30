import numpy as np
import pandas
import scipy
from scipy.io import loadmat
import math
import pandas as pd
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt

import math
import seaborn as sns
sns.set_style("whitegrid")

loadratio=1.4
m=loadmat("QuantileSolutionRampStatic161028.mat")
load=np.array(loadratio*m['netloadnw']).reshape(-1)

bounds = [1,5,10]
oris = [0]
winds = [3]
emi_type = "mc"
def get_ratio_mean(bound,ori,emi_type,windratio):
    cost_trivial = np.load("results/%.2f%s%.1fcost_hours_no_battery.npy"% (bound/100.0, emi_type,windratio))
    cost_ori_trivial = np.load("results/%.2f%s%.1fcost_hours_no_battery.npy"% (ori/100.0, emi_type,windratio))
    ratio = np.zeros(2728)
    delta = np.zeros(2728)
    ratio_mean = np.zeros(2728)
    zeros = 0
    mean = np.mean(np.array(cost_ori_trivial))
    for t in range(len(ratio)):
        # print(cost_ori_trivial[t])
        delta[t] = (cost_trivial[t]-cost_ori_trivial[t])
        ratio_mean[t] = (cost_trivial[t]-cost_ori_trivial[t])/mean
        if cost_ori_trivial[t]!=0:
            ratio[t-zeros]= (cost_trivial[t]-cost_ori_trivial[t])/cost_ori_trivial[t]
        else:
            zeros +=1
    return ratio_mean
for ori in oris:

    for bound in bounds:
        plt.figure(figsize=(8,4),dpi=150)

        # ratio_mean3 = get_ratio_mean(bound,ori,emi_type,3)
        # ratio_mean6= get_ratio_mean(bound,ori,emi_type,6.5)
        # # plt.scatter(ratio_mean3,ratio_mean6,s=2,alpha = 0.5)

        # # plt.legend()

        # # plt.xlabel('Cost Improvement ratio mean(windratio 3)',fontsize=18)
        # # plt.ylabel('Cost Improvement ratio mean(windratio 6.5)',fontsize=10)
        # # # plt.xlabel('Cost Improvement ratio mean',fontsize=18)

        # # plt.tight_layout()
        # # plt.savefig(f"figs/cost improvement ratio mean bound{bound}_base{ori}(3 vs 6.5, no battery).png")
        # # # plt.savefig(f"figs/cost improvement ratio mean_bound{bound}_base{ori}(3,6.5).png")
        # diff = ratio_mean6-ratio_mean3
        # print(ori,bound)
        # pos, neg=0,0
        # for i in range(len(diff)):
        #     if diff[i]>0:
        #         pos += diff[i]
        #     else:
        #         neg +=diff[i]
        # print("zeros", np.sum(abs(diff)<1e-6))
        # print("bigger", np.sum(diff>1e-6))
        # print("smaller", np.sum(diff<-1e-6))
        # print("pos,neg:",pos,neg)


        # plt.hist(ratio_mean6-ratio_mean3,bins = 300)
        # plt.ylabel('delta cost ratio between wind ratio 6.5 and 3',fontsize = 10)
        # plt.tight_layout()
        # plt.savefig(f"figs/delta cost improvement ratio mean bound{bound}_base{ori}(3 vs 6.5,no battery).png")
        # plt.close()
        # continue
        for windratio in winds:
            cost_trivial = np.load("results/%.2f%s%.1fcost_hours_no_battery.npy"% (bound/100.0, emi_type,windratio))
            cost_ori_trivial = np.load("results/%.2f%s%.1fcost_hours_no_battery.npy"% (ori/100.0, emi_type,windratio))
            ratio = np.zeros(2728)
            delta = np.zeros(2728)
            ratio_mean = np.zeros(2728)
            zeros = 0
            mean = np.mean(np.array(cost_ori_trivial))
            for t in range(len(ratio)):
                # print(cost_ori_trivial[t])
                delta[t] = (cost_trivial[t]-cost_ori_trivial[t])
                ratio_mean[t] = (cost_trivial[t]-cost_ori_trivial[t])/mean
                if cost_ori_trivial[t]!=0:
                	ratio[t-zeros]= (cost_trivial[t]-cost_ori_trivial[t])/cost_ori_trivial[t]
                else:
                	zeros +=1
                # if cost_trivial[t]<cost_ori_trivial[t]:
                # 	print("!")
            # a = sorted(ratio_mean,reverse = True)
            # a = sorted(delta,reverse = True)

            # print(a)
            # plt.plot(range(zeros,2728),a[:2728-zeros],label=f"windratio{windratio}")
            # plt.plot(range(2728),a,label=f"windratio{windratio}")
            # m=loadmat("QuantileSolutionRampStatic161028.mat")
    
            wind=windratio*np.array(m['wexp']).reshape(-1)
            start=6024
            length=2728
            end=start+length
            x  = load-wind
            x = x[start:end]
            # x = (wind[start+2:end]-wind[start+1:end-1])**2+(wind[start+1:end-1]-wind[start:end-2])**2
            print(x.shape)
            # print(ratio_mean.shape)
            y = ratio_mean
            print(y.shape)
            # plt.scatter(x,y,s=2,label=f"windratio{windratio}")
            # plt.hist(y,bins=200,label=f"windratio{windratio}",alpha=0.5)
            
            data = ratio_mean[:(2728//24)*24]
            data = data.reshape(-1,24)
            pd.DataFrame(data) #以一个数据框的格式来显示
            f,ax = plt.subplots(figsize=(9,6)) #定义一个子图宽高为9和6 ax存储的是图形放在哪个位置
            ax = sns.heatmap(data,center = 0)
        # plt.legend()

        # plt.xlabel('load-wind',fontsize=18)
        # plt.ylabel('Cost Improvement ratio mean',fontsize=18)
        # plt.xlabel('Cost Improvement ratio mean',fontsize=18)

        plt.tight_layout()
        # plt.savefig(f"figs/cost improvement ratio mean vs load-wind_bound{bound}_base{ori}(3,6.5).png")
        # plt.savefig(f"figs/cost improvement ratio mean_bound{bound}_base{ori}(3,battery).png")
        plt.savefig(f"figs/heatmap_bound{bound}_base{ori}(3,no battery).png")

        plt.close()


l=range(-1,11,1)
num = len(l)
error = np.load("data/error.npy").reshape(num,-1)
print(error.shape)
plt.figure(figsize=(8,4),dpi=150)
# l=[1,5,10]
length=2728
times=[]
loadratio=1.4
m=loadmat("QuantileSolutionRampStatic161028.mat")
load=np.array(loadratio*m['netloadnw']).reshape(-1)
start=6024
length=2728
end=start+length
for i,bound in enumerate(l):
    # ts=0
    # a = np.concatenate((load[bound-1,:],error[bound-1,:]),axis=0)
    
    # for j in range(length-1):
    #     if error[bound-1,j]*error[bound-1,j+1]<0:
    #         ts+=1
    # times.append(ts)
    # a = sorted(error[i], reverse =True)
    # plt.plot(a,label=f"error_bound{bound}")
    # plt.plot(error[bound-1],label=f"error_bound_{bound}-{ts}",alpha=0.5)

    plt.scatter(load[start:end],error[i],s=1,label=f"error_bound_{bound}")
plt.legend()

plt.xlabel('load',fontsize=18)
plt.ylabel('prediction error',fontsize=18)

plt.tight_layout()
plt.savefig(f"figs/error_vs_load.pdf")
plt.close()

print(times)