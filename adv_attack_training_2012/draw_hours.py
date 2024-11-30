import numpy as np
import pandas
import scipy
from scipy.io import loadmat
import math

import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt

import math
n=151
start=6024
length=2728
end=start+length
loadratio=1.4


def battery_cost_min(pred,load,wind,start,end,capacity,pe,rho,barB):
    hatB=math.floor((1-rho)*barB)
    record_emi_array=[]
    record_decision=[]
    record_theta=0
    emi_min=1e20

    temptot=0
    tempcost=0
    M = 0
    m = 1e20
    plant_emi_curve=np.array([0])
    plant_cost_curve=np.array([0])
    marginal_cost = np.array([0])
    for i in range(0,n):
        M= max(M,mc[i])
        m=min(m,mc[i])
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            tempcost=tempcost+mc[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            plant_cost_curve=np.append(plant_cost_curve,tempcost)
            marginal_cost=np.append(marginal_cost,mc[i])
            #print(plant_emi_curve)

    maxcapcity=len(plant_emi_curve)-1

    # print(M)
    print(m)
    # for i in range(1,maxcapcity+1):
    #     M = max(M, plant_emi_curve[i]/i)
    #     m = min(m, plant_emi_curve[i]/i)

    btheta=(math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2
    print("best theta:", (math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2)

    record_bat=0
    #for theta in np.arange(0,50,0.2):
    for theta in [btheta]:
        emitot=0
        bat=0

        emi_array=np.array([])
        decision=np.array([])
        for t in range(start,end,1):

            emiunit=0
            grid=0
            charging=0
            discharging=0
            w_t=wind[t]
            D_t=pred[t]

            i=0
            netD_t=D_t-w_t
            if w_t>=D_t:
                charging=min(w_t-D_t,barB-bat)
                w_t = w_t-charging
            else:
                w_t=0

            price = plant_cost_curve[1]
            netD_t=max(0,netD_t)
            if netD_t>=maxcapcity:
                print("exceed max capacity")
                print(t,D_t,w_t,netD_t)
                print(maxcapcity)
                raise ValueError
            if netD_t>0:
                price=(plant_cost_curve[netD_t+1]-plant_cost_curve[netD_t-1])/2
            # if D_t>maxcapcity:
            #     print("exceed max capacity")
            # price = plant_emi_curve[D_t]/D_t
            if price<=theta:
                i=netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
            else:
                discharging=min(netD_t,bat)
                if t==6311:
                    print(netD_t,bat)
                    print(discharging)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                emiunit=plant_emi_curve[netD_t-discharging]
            if pred[t]<load[t]:
                need = load[t]-pred[t]-w_t
                if need > 0:
                    if  discharging+need<=bat:
                        discharging=discharging+need
                    else:
                        i=i+need-(bat-discharging)
                        discharging=bat
                        emiunit += marginal_cost[i]*(need-(bat-discharging))
                    if t==6311:
                        print(pred[t],load[t],bat,i,need)
                        print(discharging)
            else:
                delta = min(barB-charging-bat,pred[t]-load[t])
                if delta<pred[t]-load[t]:
                    emiunit += marginal_cost[i]*(pred[t]-load[t]-delta)
                charging=min(barB-bat,charging+pred[t]-load[t])

            bat=bat+charging-discharging
            if charging<0 or discharging<0:
                print(t,charging,discharging)
                raise ValueError
            if charging>barB or discharging>barB:
                print(t,charging,discharging)
                raise ValueError

            decision=np.append(decision,charging-discharging)
            emi_array=np.append(emi_array,emiunit)
            #print(emiunit)
            emitot=emitot+emiunit
            # print(emitot)
        if emitot<emi_min:
            emi_min=emitot
            record_theta=theta
            # print(theta)
            # print(emitot)
            record_emi_array=emi_array
            record_bat=bat
            record_decision=decision
        #print(theta)
        # print(emi_array)
    print("minimum emission:", emi_min)
    # print("corresponding theta:", record_theta)
    # # print(record_emi_array)
    # print("corresponding final battery: ", record_bat)
    return emi_min,record_theta,record_emi_array,record_decision

def battery_trivial(pred,load,wind,start,end,capacity, pe, barB):

    emi_min=1e20

    temptot=0
    emi_plant_curve=[]
    cap_plant_curve=[]
    capcount=int(0)
    plant_emi_curve=np.array([0])
    marginal_cost = np.array([0])
    for i in range(0,n):
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            marginal_cost = np.append(marginal_cost,pe[i])
            #print(plant_emi_curve)
        capcount=capcount+capacity[i]
        emi_plant_curve=np.append(emi_plant_curve,temptot)
        cap_plant_curve=np.append(cap_plant_curve,capcount)
    cap_plant_curve=np.rint(cap_plant_curve).astype(np.int64)
    maxcapcity=len(plant_emi_curve)-1

        
    emitot=0
    bat=0

    emi_array=np.array([])
    decision=np.array([])
    prod_hour=[]
    cost_hour=[]
    for t in range(start,end,1):
            # if t%100 ==0:
            #     print(t)
        emiunit=0
        charging=0
        discharging=0
        w_t=wind[t]
        D_t=pred[t]
            # if bat>0:
            #     print(bat)
        netD_t=D_t-w_t
        if netD_t<0:
            charging=min(-netD_t,barB-bat)
            w_t -= D_t
            i=0
            netD_t=0
        else:
            w_t = 0
            discharging=min(netD_t,bat)
            netD_t=netD_t-discharging
            i = netD_t
            emiunit=plant_emi_curve[netD_t]
        if pred[t]<load[t]:
            need = load[t]-pred[t]-w_t
            if need > 0:
                if  discharging+need<=bat:
                    discharging=discharging+need
                else:
                    i=i+need-(bat-discharging)
                    discharging=bat
                    emiunit += marginal_cost[i]*(need-(bat-discharging))
        else:
            delta = min(barB-charging-bat,pred[t]-load[t])
            if delta<pred[t]-load[t]:
                emiunit += marginal_cost[i]*(pred[t]-load[t]-delta)
            charging=min(barB-bat,charging+pred[t]-load[t])

        prod_hour.append(i)
        cost_hour.append(plant_emi_curve[i])

        bat=bat+charging-discharging
        decision=np.append(decision,charging-discharging)
        emi_array=np.append(emi_array,emiunit)
        #print(emiunit)
        emitot=emitot+emiunit
        # print(emitot)

    print("minimum emission:", emitot)
    # print(record_emi_array)
    # print("corresponding final battery: ", bat)
    return emitot, emi_array, decision,np.array(prod_hour),np.array(cost_hour)





m=loadmat("QuantileSolutionRampStatic161028.mat")

netloadnw=np.array(m['netloadnw']).reshape(-1)
# netloadnw.to_csv('year_demand.csv')
dataframe=pandas.DataFrame(netloadnw)
dataframe.to_csv('year_demand.csv')
# print(netloadnw[start:end])
print(netloadnw[start:end].shape)

capacity=np.array(m['capacity']).reshape(-1)
mc=np.array(m['mc']).reshape(-1)
capacity=np.rint(capacity).astype(np.int64)
ser=np.array(m['ser']).reshape(-1)
ner=np.array(m['ner']).reshape(-1)
cer=np.array(m['cer']).reshape(-1)
cher=np.array(m['cher']).reshape(-1)
id_=np.argsort(mc)
ner=ner[id_]
ser=ser[id_]
cer=cer[id_]
cher=cher[id_]
capacity=capacity[id_]
pe = mc
emi_type = "mc"

def run(bound):
    record_improvement_battery=[]
    record_improvement_trivial=[]
    record_improvement_battery_rho=[]
    record_improvement_battery_cost=[]
    record_improvement_battery_cost_min=[]

    record_emi_no=[]
    record_emi_battery_cost_min=[]
    record_emi_trivial=[]

    record_improvement_battery_pc=[]
    record_improvement_trivial_pc=[]
    record_improvement_battery_rho_pc=[]
    record_improvement_battery_cost_pc=[]
    record_improvement_battery_cost_min_pc=[]
    windratio_array=[3,4,5,6.5]


# plt.plot(x,tri[:,id],label="battery_trivial")  
# plt.plot(x,no[:,id],label="no battery")  





    #windratio_array=[6.5]
    colors = ['r','g','b','orange']
    for iden,windratio in enumerate(windratio_array):
    #windratio=6.5

        load=np.array(loadratio*m['netloadnw']).reshape(-1)
        wind=np.array(m['wexp']).reshape(-1)
        solar=np.load("changedsolar.npy")#16000
        wind=windratio*(wind+solar)


        load=np.rint(load)
        # print(load)
        load=load.astype(np.int64)
        # print(load)
        wind=np.rint(wind).astype(np.int64)
        if bound <-0.001:
            b = load / loadratio
        elif bound <0.001:
            b = np.genfromtxt(f'results/pred_data.csv',dtype=float,delimiter=',')
            b= b[:,0].reshape(-1)
            zero=np.zeros(start).reshape(-1)
            b=np.append(zero,b)
            print(b.shape)
        else:
            print("load_attack_bound%.2f_result"%(bound))
            b = np.genfromtxt(f'results/load_attack_bound%.2f_result.csv'%(bound),dtype=float,delimiter=',')
            b= b[:,0].reshape(-1)
            zero=np.zeros(start).reshape(-1)
            b=np.append(zero,b)
            print(b.shape)
        #print(end)


        pred=np.rint(loadratio*b).astype(np.int64)


        for i in range(start,start+5):
            print(load[i],pred[i])


        r_sum=0
        a_sum=0
        for i in range(start,end):
            if wind[i]>load[i]:
                r_sum=r_sum+wind[i]-load[i]
            else:
                a_sum=a_sum+load[i]-wind[i]

        rho=r_sum/a_sum
        print("="*25,"rho = ",rho,"="*25)

        # print("*"*10,'battery to minimize cost',"*"*10)
        # [emi_battery_cost_min,theta_battery_cost_min,emi_array_battery_cost_min,decision_battery_cost_min]=battery_cost_min(pred,load,wind,start,end,capacity,pe,rho,16000)

        print("*"*10,'trivial',"*"*10)
        [emi_trivial,emi_array_trivial,decision_trivial,prod_trivial,emiunit_trivial]=battery_trivial(pred,load,wind,start,end,capacity,pe,16000)
        print("*"*10,'no battery',"*"*10)
        # [emi_no,theta_no,emi_array_no,decision_no]=battery_simple(pred,wind,start,end,capacity,pe,0.5,0.51,0.1,0)
        # print("emi_no",emi_no)
        [emi_no,emi_array_no,decision_no,prod_no,emiunit_no]=battery_trivial(pred,load,wind,start,end,capacity,pe,0)
        print("emi_no",emi_no)
        np.save("results/%.2f%s%.1fcost_hours_trivial"% (bound, emi_type,windratio), emi_array_trivial)

        np.save("results/%.2f%s%.1fcost_hours_no_battery"% (bound, emi_type,windratio), emi_array_no)
        # plt.figure(figsize=(8,4),dpi=150)

        # a = sorted(emi_array_trivial-,reverse=True)
        # plt.plot(a,color = colors[iden],label=f"rho_{windratio}",alpha=0.5)
        # plt.plot(emi_array_no,color = 'purple',label=f"rho_{windratio}-no_battery",linestyle='dashed',alpha=0.5)


        # plt.figure(figsize=(8,4),dpi=150)

        # plt.plot(emiunit_trivial,color = colors[iden],label=f"rho_{windratio}",alpha=0.5)
        # plt.plot(emiunit_no,color = 'purple',label=f"rho_{windratio}-no_battery",linestyle='dashed',alpha=0.5)
        # plt.legend()

        # plt.xlabel('hour',fontsize=18)
        # plt.ylabel('Cost',fontsize=18)

        # plt.tight_layout()
        # plt.savefig(f"figs/Cost_{bound}-windratio_{windratio}.png")
        # plt.close()

        # plt.figure(figsize=(8,4),dpi=150)

        # plt.plot(prod_trivial,color = colors[iden],label=f"rho_{windratio}",alpha=0.5)
        # plt.plot(prod_no,color = 'purple',label=f"rho_{windratio}-no_battery",linestyle='dashed',alpha=0.5)
        # plt.legend()

        # plt.xlabel('hour',fontsize=18)
        # plt.ylabel('Product',fontsize=18)

        # plt.tight_layout()
        # plt.savefig(f"figs/Product_{bound}-windratio_{windratio}.png")
        # plt.close()

    # plt.legend()

    # plt.xlabel('hour',fontsize=18)
    # plt.ylabel('Cost',fontsize=18)

    # plt.tight_layout()
    # plt.savefig(f"figs/TrueCost_{bound}.png")
    # plt.close()
        print('emi_no%.1f='% (windratio),emi_no)
        print('emi_trivial%.1f='% (windratio),emi_trivial)
        # print('emi_battery_cost_min%.1f='% (windratio),emi_battery_cost_min)
        record_emi_no.append(emi_no)
        # record_emi_battery_cost_min.append(emi_battery_cost_min)
        record_emi_trivial.append(emi_trivial)

        # print('emi_battery_cost_min_improve%.1f='% (windratio),1-emi_battery_cost_min/emi_no)
        print('emi_trivial_improve%.1f='% (windratio),1-emi_trivial/emi_no)

        # print('theta_battery_cost_min%.1f='% (windratio),theta_battery_cost_min)

        # record_improvement_battery_cost_min.append(emi_no-emi_battery_cost_min)
        record_improvement_trivial.append(emi_no-emi_trivial)


        # record_improvement_battery_cost_min_pc.append(1-emi_battery_cost_min/emi_no)
        record_improvement_trivial_pc.append(1-emi_trivial/emi_no)

        # np.save("results/%.2f%s%.1fwind_emi_array_battery_cost_min"% (bound, emi_type,windratio), emi_array_battery_cost_min)
        np.save("results/%.2f%s%.1fwind_emi_array_no"% (bound, emi_type,windratio), emi_array_no)
        np.save("results/%.2f%s%.1fwind_emi_array_trivial"% (bound, emi_type,windratio), emi_array_trivial)

        # np.save("results/%.2f%s%.1fwind_decision_battery_cost_min"% (bound, emi_type,windratio), decision_battery_cost_min)
        np.save("results/%.2f%s%.1fwind_decision_trivial"% (bound, emi_type,windratio), decision_trivial)

        np.save("results/%.2f%s%.1fwind_netload_to_print"% (bound, emi_type,windratio), load-wind)

    # record_improvement_battery_cost_min=np.array(record_improvement_battery_cost_min)
    record_improvement_trivial=np.array(record_improvement_trivial)

    # record_improvement_battery_cost_min_pc=np.array(record_improvement_battery_cost_min_pc)
    record_improvement_trivial_pc=np.array(record_improvement_trivial_pc)

    record_emi_no=np.array(record_emi_no)
    # record_emi_battery_cost_min=np.array(record_emi_battery_cost_min)
    record_emi_trivial=np.array(record_emi_trivial)

    # np.save("results/%.2f%semi_battery_cost_min"% (bound, emi_type), record_emi_battery_cost_min)
    np.save("results/%.2f%semi_trivial"% (bound, emi_type), record_emi_trivial)
    np.save("results/%.2f%semi_no"% (bound, emi_type), record_emi_no)

    # np.save("results/%.2f%simprovement_battery_cost_min"% (bound, emi_type), record_improvement_battery_cost_min)
    np.save("results/%.2f%simprovement_trivial"% (bound, emi_type), record_improvement_trivial)

    # np.save("results/%.2f%simprovement_pc_battery_cost_min"% (bound, emi_type), record_improvement_battery_cost_min_pc)
    np.save("results/%.2f%simprovement_pc_trivial"% (bound, emi_type), record_improvement_trivial_pc)
    return load[start:end],load[start:end]-pred[start:end]


import seaborn as sns
import argparse

sns.set_style("whitegrid")

idx = range(-1,11,1)
error = np.array([])
load= np.array([])
times = []
for i,bound in enumerate(idx):
    l,e = run(bound/100.0)
    load = np.concatenate((load,l),axis=0)
    error = np.concatenate((error,e),axis=0)
# load=np.array(load)
# error=np.array(error)
np.save("data/error",error)
np.save("data/load",load)
plt.figure(figsize=(8,4),dpi=150)

for i,bound in enumerate(l):
    ts=0
    for j in range(2728-1):
        if error[i,j]*error[i,j+1]<0:
            ts+=1
    times.append(ts)
    plt.plot(error[i],label=f"error_bound_{bound}-{ts}",alpha=0.5)

plt.legend()

plt.xlabel('hour',fontsize=18)
plt.ylabel('prediction error',fontsize=18)

plt.tight_layout()
plt.savefig(f"figs/error_4.png")
plt.close()

print(times)
import pdb; pdb.set_trace()
