import numpy as np
import pandas
import scipy
from scipy.io import loadmat
import math
import copy
import time
import ipdb
from numba import jit, njit
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
np.random.seed(11)

"""
start=6024
length=2728
end=start+length
loadratio=1.4
"""
m=loadmat("QuantileSolutionRampStatic161028.mat")

mc=np.array(m['mc']).reshape(-1)

def sample_from(distribution):
    return np.random.choice(distribution.shape[0], p = distribution)
    
def battery_cost_integral(pred,load,wind,start,end,capacity,pe,rho,barB,mean_net_demand):
    #hatb, mn theta, trivial first
    # hatB=math.floor((1-rho)*barB)
    hatB = barB
    record_emi_array=[]
    record_decision=[]
    record_theta=[]
    emi_min=1e20

    temptot=0
    tempcost=0
    plant_emi_curve=np.array([0])
    plant_cost_curve=np.array([0])
    marginal_cost = np.array([0])
    power_plant = np.array([0])
    mc = pe
    for i in range(0,capacity.shape[0]):
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            tempcost=tempcost+mc[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            plant_cost_curve=np.append(plant_cost_curve,tempcost)
            marginal_cost=np.append(marginal_cost,mc[i])
        power_plant=np.append(power_plant, power_plant[-1] + capacity[i])

    maxcapcity=len(plant_emi_curve)-1

    def heuristic(cost, bat):
        # print(max(0,bat-np.floor(rho*mean_net_demand)),bat, cost)
        return cost
        # return cost - plant_emi_curve[max(0,bat//2)]
        # return cost - bat

    def update(id_theta,theta, w_t, D_t, l_t, bat):
        emiunit=0
        charging=0
        discharging=0
        discharging_delta = 0
        decision = 0
        i=0
        #different from trivial, we consider the cost to abandon the power, so when wind is not enough, we first let plants to produce power, not let battery discharges.
        netD_t=D_t-w_t
        if w_t>=D_t:
            charging=min(w_t-D_t,barB-bat)
            # w_t = w_t-charging
            netw_t = w_t - D_t - charging
        else:
            # w_t=0
            netw_t = 0
            discharging_delta = min(netD_t,bat)
            netD_t=netD_t-discharging_delta
            bat = bat - discharging_delta
            decision = -discharging_delta
        netD_t=max(0,netD_t)

        price = marginal_cost[netD_t]
        if netD_t>=maxcapcity:
            print("exceed max capacity")
            print(t,D_t,w_t,netD_t)
            print(maxcapcity)
            raise ValueError

        '''
        non-convex mechanism
        '''
        if price<=theta:
            i=netD_t+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))
            emiunit=plant_emi_curve[netD_t+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))]
            charging=charging+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))
        else:
            discharging=min(netD_t-power_plant[id_theta],bat)
            i=netD_t-discharging
            if i>maxcapcity:
                print(t,netD_t,D_t,w_t,discharging)
                print(maxcapcity)
                raise ValueError
            emiunit=plant_emi_curve[netD_t-discharging]
        
        netcharge = charging - discharging
        
        if D_t<l_t:
            need = l_t-D_t-netw_t
            if need > 0:
                if need <= bat + netcharge:
                    netcharge -= need 
                else:
                    i=i+need-(bat+netcharge)
                    emiunit += marginal_cost[i]*(need-(bat+netcharge))
                    netcharge = -bat
            else:
                if netcharge+bat<barB or netcharge+bat>barB:
                    raise ValueError
                delta = min(-need, barB-bat-netcharge)
                if -need > barB-bat-netcharge:
                    emiunit += marginal_cost[i]*(-need - (barB-bat-netcharge))
                    if marginal_cost[i]*(-need - (barB-bat-netcharge)) != 0:
                        raise ValueError
                netcharge += delta
        else:
            delta = min(barB-netcharge-bat-netw_t,D_t-l_t)
            if delta<D_t-l_t:
                emiunit += marginal_cost[i]*(D_t-l_t-delta)
            netcharge += min(barB-netcharge-bat,D_t-l_t+netw_t)

        assert i+w_t-netcharge+discharging_delta>=l_t
        decision += netcharge
        if netcharge<-bat or bat+netcharge >barB:
            print(t,charging,discharging)
            import ipdb;ipdb.set_trace()
            raise ValueError

        bat=bat+netcharge

        return emiunit, bat, decision

    theta_list = np.concatenate(([0],mc))+1e-8
    id_theta_list = np.arange(theta_list.shape[0])
    # id_theta_list= np.array([0])
    theta_list = theta_list[id_theta_list]
    bat_list = np.zeros(theta_list.shape[0]).astype(np.int32)
    # pseudo_cost = np.zeros_like(theta_list)
    cost_list = np.zeros((theta_list.shape[0],end-start))
    bat = 0
    emitot = 0
    w = np.ones_like(theta_list)
    eta = 800*np.sqrt(np.log(theta_list.shape[0])/(end-start))
    print("eta",eta)
    id_theta_t = 0
    for t in range(start,end,1):
        # print(t)
        # if t-start >668:
        #     print(bat)
        # if t-start == 738:
        #     ipdb.set_trace()
        probability = w/np.sum(w)
        p = np.unique(probability)
        if (t-start) % 5 ==0 or (p.shape[0]!=1 and np.max(p)-np.partition(p, -2)[-2]>0.1):
            # if (t-start)%10 != 0:
            #     print("exceed diff prob")
            id_theta_t = sample_from(probability)
            bat_list = np.ones(theta_list.shape[0]).astype(np.int32)*bat
        emiunit, bat_return, decision = update(id_theta_list[id_theta_t], theta_list[id_theta_t], wind[t], pred[t], load[t], bat)
        record_decision.append(decision)
        emitot += emiunit
        record_theta.append(theta_list[id_theta_t])
        record_emi_array.append(emiunit)
        #print(theta)
        # ipdb.set_trace()
        for id_theta_, id_theta in enumerate(id_theta_list):
            theta = theta_list[id_theta_]
        # for id_theta,theta in enumerate(theta_list):
            # print(bat_list[id_theta_])
            emiunt, bat_r, _ = update(id_theta, theta, wind[t], pred[t], load[t], bat_list[id_theta_])
            cost = heuristic(emiunt, bat_r)
            bat_list[id_theta_] = bat_r
            cost_list[id_theta_][t-start] = cost
            cost = cost / plant_emi_curve[-1]
            #multiplicative weights
            w[id_theta_] = w[id_theta_]*np.exp(-eta*cost)
        bat = bat_return

    # print("minimum emission:", emitot)
    # print("corresponding theta:", record_theta[-10:])

    # plt.clf()
    # plt.plot(record_emi_array, label=f"mw:{emitot}",alpha=0.2)
    # plt.plot(cost_list[0],label=f"0:{np.sum(cost_list[0])}",alpha=0.2)
    # plt.legend()
    # plt.savefig("debug.jpg")
    # ipdb.set_trace()
    return emitot,record_theta,record_emi_array,record_decision


# @njit
# def check_emi_decision(plant_emi_curve, marginal_cost, wind, pred, load, barB, decision, start, end):
#     bat=0
#     emi_array=np.zeros(end-start)
#     emitot = 0
#     #print("start", start)
#     #print("end", end)
#     for t in range(start,end):
#         emiunit=0
#         # print(t)
#         w_t=wind[t]
#         D_t=pred[t]
#         netD_t=D_t-w_t
#         # print(netD_t)
#         charging, discharging = 0, 0
#         if netD_t<0:
#             charging=min(-netD_t,barB-bat)
#             w_t -= D_t+charging
#             i=0
#             netD_t=0
#         else:
#             w_t = 0
#             discharging=min(netD_t,bat)
#             netD_t=round(netD_t-discharging)
#             i = netD_t
#             emiunit=plant_emi_curve[netD_t]
#             # print(netD_t)
#         need = load[t]-pred[t]-w_t + decision[t-start] - (charging - discharging)
#         #print("need", need)
#         #print(load[t])
#         #print(pred[t])
#         #print(w_t)
#         #print(decision[t-start])
#         #print(charging - discharging)
#         if need > 0:
#             i=i+need
#             emiunit += marginal_cost[round(i)]*(need)
#             #print(int(i))
#         else:
#             emiunit += marginal_cost[round(i)]*(-need)
#             #print(int(i))
#         if decision[t-start] > charging - discharging:
#             charging += decision[t-start] - (charging - discharging)
#         else:
#             discharging += charging - discharging - decision[t-start]

#         bat=bat+charging-discharging
#         emi_array[t-start]= emiunit
#         # print(t,emiunit)
#         emitot=emitot+emiunit

#     return emitot, emi_array

@njit
def check_emi_decision_1(plant_emi_curve, marginal_cost, wind, pred, load, barB, decision, start, end, online):
    bat=0
    emi_array=np.zeros(end-start)
    emitot = 0
    #print("start", start)
    #print("end", end)
    for t in range(start,end):
        emiunit=0
        # print(t)
        w_t=wind[t]
        D_t=pred[t]
        netD_t=D_t-w_t
        ifcharge = decision[t-start]
        p2 = load[t] + ifcharge - wind[t] 
        if online == False:
            if max(0,p2) + wind[t] + bat >= D_t:
                p1 = max(0,p2)
                emiunit = plant_emi_curve[round(max(p2,0))]
            else:
                p1 = D_t - wind[t] -bat
                emiunit = plant_emi_curve[round(p1)] + marginal_cost[round(max(p1,p2))]*abs(p1-p2)
        else:
            p1 = max(0,D_t - wind[t] -bat)
            emiunit = plant_emi_curve[round(p1)] + marginal_cost[round(max(p1,p2))]*abs(p1-p2)
        # print(p1,p2)
        # print(netD_t)
        # charging, discharging = 0, 0
        # if netD_t<0:
        #     charging=min(-netD_t,barB-bat)
        #     w_t -= D_t+charging
        #     i=0
        #     netD_t=0
        # else:
        #     w_t = 0
        #     discharging=min(netD_t,bat)
        #     netD_t=round(netD_t-discharging)
        #     i = netD_t
        #     emiunit=plant_emi_curve[netD_t]
        #     # print(netD_t)
        # need = load[t]-pred[t]-w_t + decision[t-start] - (charging - discharging)
        # #print("need", need)
        # #print(load[t])
        # #print(pred[t])
        # #print(w_t)
        # #print(decision[t-start])
        # #print(charging - discharging)
        # if need > 0:
        #     i=i+need
        #     emiunit += marginal_cost[round(i)]*(need)
        #     #print(int(i))
        # else:
        #     emiunit += marginal_cost[round(i)]*(-need)
        #     #print(int(i))
        # if decision[t-start] > charging - discharging:
        #     charging += decision[t-start] - (charging - discharging)
        # else:
        #     discharging += charging - discharging - decision[t-start]

        # bat=bat+charging-discharging
        bat = bat+ifcharge
        emi_array[t-start]= emiunit
        # print(t,emiunit)
        emitot=emitot+emiunit

    return emitot, emi_array


# @njit
# def battery_opt(pred,load,wind,start,end,capacity, pe, barB):
#     # barB=2
#     emi_min=1e20
#     temptot=0
#     # emi_plant_curve=[]
#     # cap_plant_curve=[]
#     # capcount=int(0)
#     plant_emi_curve=np.array([0], dtype=np.float64)
#     marginal_cost = np.array([0], dtype=np.float64)
#     maxcapacity = 0
#     for i in range(0,n):
#         maxcapacity += capacity[i]
#         for j in range(0,capacity[i]):
#             temptot=temptot+pe[i]
#             plant_emi_curve=np.append(plant_emi_curve,temptot)
#             marginal_cost = np.append(marginal_cost,pe[i])
#             #print(plant_emi_curve)
#         # capcount=capcount+capacity[i]
#     #     emi_plant_curve=np.append(emi_plant_curve,temptot)
#     #     cap_plant_curve=np.append(cap_plant_curve,capcount)
#     # cap_plant_curve=np.rint(cap_plant_curve).astype(np.int64)
#     # maxcapacity = plant_emi_curve.shape[0] -1

#     # emi_bat = np.zeros((end+1,barB+1))
#     emi_bat_t1 =np.zeros(barB+1)
#     emi_bat_t0 = np.zeros(barB+1)
#     record_decision = np.zeros((end-start,barB+1))
#     # temp_emi = np.zeros((end-start,barB+1))
#     # time1,time2,time3,time4=0,0,0,0
#     for t in range(end,start,-1):
#         D_t = pred[t-1]
#         l_t = load[t-1]
#         if t % 500 == 0:
#             print(t)
#         # print(t-start-1,emi_bat_t0[0])
#         for bat_temp in range(barB+1):
#             emi_bat_t1[bat_temp] = emi_bat_t0[bat_temp]
#             emi_bat_t0[bat_temp] = emi_min
#             # temp_emi[t-start-1][bat_temp] = emi_min
#         # print(t-start-1,emi_bat_t0[0],emi_bat_t1[0])

#         # emi_bat_t1 =copy.deepcopy(emi_bat_t0)
#         # emi_bat_t0 = np.zeros(barB+1)
#         for bat_last in range(barB+1):
#             # emi_bat[t-1][bat_last] = emi_min
#             # emi_bat_t0[bat_last] = emi_min
#             for bat_cur in range(barB+1):
#                 ifcharge = bat_cur - bat_last
#                 bat = bat_last
#                 emi_temp = 0
#                 w_t = wind[t-1]
#                 # D_t = pred[t-1]
#                 # l_t = load[t-1]
#                 netD_t = D_t - w_t
#                 #last_clock = time.time()

#                 if netD_t < 0:
#                     charging = min(-netD_t, barB - bat)
#                     w_t -= D_t + charging
#                     i=0
#                     netD_t = 0
#                 else:
#                     w_t = 0
#                     charging = - min(netD_t, bat)
#                     netD_t += charging
#                     i = netD_t
#                     emi_temp = plant_emi_curve[netD_t]
                
#                 #time1 += time.time() - last_clock
#                 #last_clock = time.time()
#                 need = l_t - D_t - w_t + ifcharge - charging
                
#                 #time2 += time.time() - last_clock
#                 #last_clock = time.time()   
                            
#                 if need > 0:
#                     i = i + need
#                     if i > maxcapacity:
#                         print(i,maxcapacity)
#                         raise ValueError
#                     emi_temp += marginal_cost[i]*need
#                 else:
#                     emi_temp += marginal_cost[i]*(-need)
                    
#                 #time3 += time.time() - last_clock
#                 #last_clock = time.time()
#                 if emi_temp < 0:
#                     raise ValueError
#                 if emi_temp + emi_bat_t1[bat_cur] < emi_bat_t0[bat_last]:
#                     emi_bat_t0[bat_last] = emi_temp + emi_bat_t1[bat_cur]
#                     record_decision[t-start-1][bat_last] = bat_cur
#                     # print(emi_bat_t1[bat_cur])
#                     # temp_emi[t-start-1][bat_last] = emi_temp + emi_bat_t1[bat_cur]
#         # print(t-start-1,emi_bat_t0[0])
#                 # emi_bat[t-1][bat_last] = min(emi_bat[t-1][bat_last], emi_temp + emi_bat[t][bat_cur])
#                 #time4 += time.time() - last_clock
#                 #last_clock = time.time()
#         # temp  = [0,1453]
#         # print(wind[t-1],load[t-1],pred[t-1],emi_bat_t0[0],emi_bat_t0[1453])

#     decision = np.zeros(end-start)
#     bat = round(0)
#     for t in range(start,end):
#         nxt = record_decision[t-start][bat]
#         # if t-start+1<end:
#         #     print(t-start, temp_emi[t-start][bat],temp_emi[t-start+1][round(nxt)],bat,nxt)
#         decision[t-start] = nxt - bat
#         bat = round(nxt)
#     # print(emi_bat_t0[0])
#     # print(decision)
#     # raise ValueError
#     # emi_array = np.zeros(end-start)
#     emi_tot, emi_array = check_emi_decision(plant_emi_curve, marginal_cost, wind, pred, load, barB, decision, start, end)
#     # bat=0
#     # emi_array=np.array(end-start)
#     # emi_tot=0
#     # for t in range(start,end):
#     #     emiunit=0
#     #     charging, discharging = 0, 0
#     #     w_t=wind[t]
#     #     D_t=pred[t]
#     #     netD_t=D_t-w_t
#     #     if netD_t<0:
#     #         charging=min(-netD_t,barB-bat)
#     #         w_t -= D_t
#     #         i=0
#     #         netD_t=0
#     #     else:
#     #         w_t = 0
#     #         discharging=min(netD_t,bat)
#     #         netD_t=netD_t-discharging
#     #         i = netD_t
#     #         emiunit=plant_emi_curve[netD_t]
#     #     need = load[t]-pred[t]-w_t + decision[t-start] - (charging - discharging)
#     #     if need > 0:
#     #         i=i+need
#     #         emiunit += marginal_cost[i]*(need)
#     #     else:
#     #         emiunit += marginal_cost[i]*(-need)
#     #         i += need
#     #     if decision[t-start] > charging - discharging:
#     #         charging += decision[t-start] - (charging - discharging)
#     #     else:
#     #         discharging += charging - discharging - decision[t-start]

#     #     bat=bat+charging-discharging
#     #     emi_array[t-start]= emiunit
#     #     emi_tot+=emiunit
#     print(emi_tot, emi_bat_t0[0])
#     assert abs(emi_tot - emi_bat_t0[0])/emi_tot < 1e-12
#     return emi_bat_t0[0], emi_array, decision


@njit
def battery_opt_1(pred,load,wind,start,end,capacity, pe, barB, online = False):
    # barB=2
    emi_min=1e20
    temptot=0
    # emi_plant_curve=[]
    # cap_plant_curve=[]
    # capcount=int(0)
    plant_emi_curve=np.array([0], dtype=np.float64)
    marginal_cost = np.array([0], dtype=np.float64)
    maxcapacity = 0
    for i in range(0,capacity.shape[0]):
        maxcapacity += capacity[i]
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            marginal_cost = np.append(marginal_cost,pe[i])
            #print(plant_emi_curve)
        # capcount=capcount+capacity[i]
    #     emi_plant_curve=np.append(emi_plant_curve,temptot)
    #     cap_plant_curve=np.append(cap_plant_curve,capcount)
    # cap_plant_curve=np.rint(cap_plant_curve).astype(np.int64)
    # maxcapacity = plant_emi_curve.shape[0] -1

    # emi_bat = np.zeros((end+1,barB+1))
    emi_bat_t1 =np.zeros(barB+1)
    emi_bat_t0 = np.zeros(barB+1)
    record_decision = np.zeros((end-start,barB+1))
    # temp_emi = np.zeros((end-start,barB+1))
    # time1,time2,time3,time4=0,0,0,0
    for t in range(end,start,-1):
        D_t = pred[t-1]
        l_t = load[t-1]
        if t % 500 == 0:
            print(t)
        # print(t-start-1,emi_bat_t0[0])
        for bat_temp in range(barB+1):
            emi_bat_t1[bat_temp] = emi_bat_t0[bat_temp]
            emi_bat_t0[bat_temp] = emi_min
            # temp_emi[t-start-1][bat_temp] = emi_min
        # print(t-start-1,emi_bat_t0[0],emi_bat_t1[0])

        # emi_bat_t1 =copy.deepcopy(emi_bat_t0)
        # emi_bat_t0 = np.zeros(barB+1)
        for bat_last in range(barB+1):
            # emi_bat[t-1][bat_last] = emi_min
            # emi_bat_t0[bat_last] = emi_min
            for bat_cur in range(barB+1):
                ifcharge = bat_cur - bat_last # the amount to charge
                bat = bat_last
                emi_temp = 0
                w_t = wind[t-1]
                # D_t = pred[t-1]
                # l_t = load[t-1]
                netD_t = D_t - w_t
                #last_clock = time.time()
                p2 = l_t - w_t + ifcharge 
                if online == False:
                    if max(p2,0) > D_t - w_t - bat:
                        p1 = max(p2,0)
                    else:
                        p1 = D_t - w_t - bat
                else:
                    p1 = max(0,D_t - w_t - bat)
                emi_temp = plant_emi_curve[p1]
                i = p1
                emi_temp += marginal_cost[max(p1,p2)]*abs(p1-p2)
                #time1 += time.time() - last_clock
                #last_clock = time.time()
                # need = l_t - D_t - w_t + ifcharge - charging
                
                # #time2 += time.time() - last_clock
                # #last_clock = time.time()   
                            
                # if need > 0:
                #     i = i + need
                #     if i > maxcapacity:
                #         print(i,maxcapacity)
                #         raise ValueError
                #     emi_temp += marginal_cost[i]*need
                # else:
                #     emi_temp += marginal_cost[i]*(-need)
                    
                #time3 += time.time() - last_clock
                #last_clock = time.time()
                if emi_temp < 0:
                    raise ValueError
                if emi_temp + emi_bat_t1[bat_cur] < emi_bat_t0[bat_last]:
                    emi_bat_t0[bat_last] = emi_temp + emi_bat_t1[bat_cur]
                    record_decision[t-start-1][bat_last] = bat_cur
                    # print(emi_bat_t1[bat_cur])
                    # temp_emi[t-start-1][bat_last] = emi_temp + emi_bat_t1[bat_cur]
        # print(t-start-1,emi_bat_t0[0])
                # emi_bat[t-1][bat_last] = min(emi_bat[t-1][bat_last], emi_temp + emi_bat[t][bat_cur])
                #time4 += time.time() - last_clock
                #last_clock = time.time()
        # temp  = [0,1453]
        # print(wind[t-1],load[t-1],pred[t-1],emi_bat_t0[0],emi_bat_t0[1453])

    decision = np.zeros(end-start)
    bat = round(0)
    for t in range(start,end):
        nxt = record_decision[t-start][bat]
        # if t-start+1<end:
        #     print(t-start, temp_emi[t-start][bat],temp_emi[t-start+1][round(nxt)],bat,nxt)
        decision[t-start] = nxt - bat
        bat = round(nxt)
    # print(emi_bat_t0[0])
    # print(decision)
    # raise ValueError
    # emi_array = np.zeros(end-start)
    emi_tot, emi_array = check_emi_decision_1(plant_emi_curve, marginal_cost, wind, pred, load, barB, decision, start, end, online)
    # bat=0
    # emi_array=np.array(end-start)
    # emi_tot=0
    # for t in range(start,end):
    #     emiunit=0
    #     charging, discharging = 0, 0
    #     w_t=wind[t]
    #     D_t=pred[t]
    #     netD_t=D_t-w_t
    #     if netD_t<0:
    #         charging=min(-netD_t,barB-bat)
    #         w_t -= D_t
    #         i=0
    #         netD_t=0
    #     else:
    #         w_t = 0
    #         discharging=min(netD_t,bat)
    #         netD_t=netD_t-discharging
    #         i = netD_t
    #         emiunit=plant_emi_curve[netD_t]
    #     need = load[t]-pred[t]-w_t + decision[t-start] - (charging - discharging)
    #     if need > 0:
    #         i=i+need
    #         emiunit += marginal_cost[i]*(need)
    #     else:
    #         emiunit += marginal_cost[i]*(-need)
    #         i += need
    #     if decision[t-start] > charging - discharging:
    #         charging += decision[t-start] - (charging - discharging)
    #     else:
    #         discharging += charging - discharging - decision[t-start]

    #     bat=bat+charging-discharging
    #     emi_array[t-start]= emiunit
    #     emi_tot+=emiunit
    print(emi_tot, emi_bat_t0[0])
    assert abs(emi_tot - emi_bat_t0[0])/emi_tot < 1e-12
    return emi_bat_t0[0], emi_array, decision






def battery_cost_min(pred,load,wind,start,end,capacity,pe,rho,barB, theoretical_best):
    # hatB=math.floor((1-rho)*barB)
    hatB = barB
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
    power_plant = np.array([0])
    for i in range(0,capacity.shape[0]):
        M= max(M,mc[i])
        m=min(m,mc[i])
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            tempcost=tempcost+mc[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            plant_cost_curve=np.append(plant_cost_curve,tempcost)
            marginal_cost=np.append(marginal_cost,mc[i])
        power_plant=np.append(power_plant, power_plant[-1] + capacity[i])
            #print(plant_emi_curve)

    maxcapcity=len(plant_emi_curve)-1

    # print(M)
    # print(m)
    # for i in range(1,maxcapcity+1):
    #     M = max(M, plant_emi_curve[i]/i)
    #     m = min(m, plant_emi_curve[i]/i)
    if theoretical_best:
        '''
        minghua chen
        '''
        '''
        btheta=(math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2
        print("best theta:", (math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2)
        '''
        theta_mean = math.sqrt(M*m)
        cost_tot, cap_tot = 0,0
        for temp_id in range(0,capacity.shape[0]):
            cost_tot += mc[temp_id] * capacity[temp_id]
            cap_tot += capacity[temp_id]
            if cost_tot/cap_tot > theta_mean:
                #with temp_id power plant, average cost is not below thetamean
                #the last one is temp_id - 1 power plant
                print(temp_id, cost_tot/cap_tot, theta_mean)
                if temp_id == 0:
                    btheta = 1e-8
                else:
                    btheta = mc[temp_id - 1] + 1e-8
                break
        theta_list = [btheta]
        id_theta_theo = 0
        while mc[id_theta_theo]<btheta:
            id_theta_theo += 1
    else:
        theta_list = np.concatenate(([0],mc))+1e-8
        # theta_list = [1e-8]
    record_bat=0
    #for theta in np.arange(0,50,0.2):
    for id_theta,theta in enumerate(theta_list):
        # print(theta,theta_list)
        if theoretical_best:
            id_theta = id_theta_theo
        emitot=0
        bat=0

        emi_array=np.array([])
        decision=np.array([])
        for t in range(start,end,1):

            emiunit=0
            grid=0
            charging=0
            discharging=0
            discharging_delta=0
            w_t=wind[t]
            D_t=pred[t]

            i=0
            #different from trivial, we consider the cost to abandon the power, so when wind is not enough, we first let plants to produce power, not let battery discharges.
            netD_t=D_t-w_t
            if w_t>=D_t:
                charging=min(w_t-D_t,barB-bat)
                # w_t = w_t-charging
                netw_t = w_t - D_t - charging
            else:
                # w_t=0
                netw_t = 0
            netD_t=max(0,netD_t)

            price = marginal_cost[netD_t]
            if netD_t>=maxcapcity:
                print("exceed max capacity")
                print(t,D_t,w_t,netD_t)
                print(maxcapcity)
                raise ValueError
            # if netD_t>0:
                # price=(plant_cost_curve[netD_t+1]-plant_cost_curve[netD_t-1])/2
            # if D_t>maxcapcity:
            #     print("exceed max capacity")
            # price = plant_emi_curve[D_t]/D_t
            '''
            minghua chen's mechanism
            '''
            ''' 
            if price<=theta:
                i=netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
            else:
                discharging=min(netD_t,bat)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                emiunit=plant_emi_curve[netD_t-discharging]
            '''

            '''
            non-convex mechanism
            '''
            if price<=theta:
                i=netD_t+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))
            else:
                discharging=min(netD_t-power_plant[id_theta],bat)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                    raise ValueError
                emiunit=plant_emi_curve[netD_t-discharging]
            
            netcharge = charging - discharging

            if pred[t]<load[t]:
                need = load[t]-pred[t]-netw_t
                if need > 0:
                    # if  discharging+need<=bat:
                    if need <= bat + netcharge:
                        netcharge -= need 
                        # discharging=discharging+need
                    else:
                        i=i+need-(bat+netcharge)
                        emiunit += marginal_cost[i]*(need-(bat+netcharge))
                        netcharge = -bat
                else:
                    if netcharge+bat<barB or netcharge+bat>barB:
                        raise ValueError
                    delta = min(-need, barB-bat-netcharge)
                    if -need > barB-bat-netcharge:
                        emiunit += marginal_cost[i]*(-need - (barB-bat-netcharge))
                        if marginal_cost[i]*(-need - (barB-bat-netcharge)) != 0:
                            raise ValueError
                    netcharge += delta
            else:
                delta = min(barB-netcharge-bat-netw_t,pred[t]-load[t])
                if delta<pred[t]-load[t]:
                    emiunit += marginal_cost[i]*(pred[t]-load[t]-delta)
                netcharge += min(barB-netcharge-bat,pred[t]-load[t]+netw_t)

            assert i+wind[t]-netcharge>=load[t]

            if netcharge<-bat or bat+netcharge >barB:
                print(t,charging,discharging)
                import ipdb;ipdb.set_trace()
                raise ValueError

            bat=bat+netcharge

            # if charging>barB or discharging>barB:
            #     print(t,charging,discharging)
            #     raise ValueError

            decision=np.append(decision,netcharge)
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
    print("corresponding theta:", record_theta)
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
    for i in range(0,capacity.shape[0]):
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
    for t in range(start,end,1):
            # if t%100 ==0:
            #     print(t)
        # import ipdb;ipdb.set_trace()
        emiunit=0
        charging=0
        discharging=0
        discharging_delta=0
        w_t=wind[t]
        D_t=pred[t]
            # if bat>0:
            #     print(bat)
        # if t-start >668:
        #     print(bat)
        # if t-start == 738:
        #     import ipdb;ipdb.set_trace()
        netD_t=D_t-w_t
        if netD_t<0:
            # charging=min(-netD_t,barB-bat)
            netw_t =w_t- D_t
            i=0
            netD_t=0
        else:
            netw_t = 0
            discharging=min(netD_t,bat)
            netD_t=netD_t-discharging
            i = netD_t
            emiunit=plant_emi_curve[netD_t]
        if pred[t]<load[t]:
            need = load[t]-pred[t]-netw_t
            if need > 0:
                if  discharging+need<=bat:
                    discharging=discharging+need
                else:
                    i=i+need-(bat-discharging)
                    emiunit += marginal_cost[i]*(need-(bat-discharging))
                    discharging=bat
            else:
                charging =min(barB-bat, netw_t - (load[t]-pred[t]))
                if barB-bat < netw_t - (load[t]-pred[t]):
                    emiunit += marginal_cost[i] * (netw_t - (load[t]-pred[t]) - (barB-bat))
                    if marginal_cost[i] * (netw_t - (load[t]-pred[t]) - (barB-bat)) != 0:
                        raise ValueError
        else:
            delta = min(discharging+barB-bat-netw_t,pred[t]-load[t])
            if delta<pred[t]-load[t]:
                emiunit += marginal_cost[i]*(pred[t]-load[t]-delta)
            charging=min(discharging+barB-bat,netw_t+pred[t]-load[t])
        if (i+wind[t]-charging+discharging< load[t]):
            import ipdb;ipdb.set_trace()
        assert i+wind[t]-charging+discharging>=load[t]
        bat=bat+charging-discharging
        decision=np.append(decision,charging-discharging)
        emi_array=np.append(emi_array,emiunit)
        # print(wind[t],load[t],pred[t],bat, charging-discharging, emiunit)
        emitot=emitot+emiunit
        # print(emitot)

    print("minimum emission:", emitot)
    # print(record_emi_array)
    # print("corresponding final battery: ", bat)
    return emitot, emi_array, decision



def battery_trivial_secondphase(pred,load,wind,start,end,capacity, pe, barB):

    emi_min=1e20

    temptot=0
    emi_plant_curve=[]
    cap_plant_curve=[]
    capcount=int(0)
    plant_emi_curve=np.array([0])
    marginal_cost = np.array([0])
    for i in range(0,capacity.shape[0]):
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
    for t in range(start,end,1):
            # if t%100 ==0:
            #     print(t)
        # import ipdb;ipdb.set_trace()
        emiunit=0
        charging=0
        discharging=0
        discharging_delta=0
        w_t=wind[t]
        D_t=pred[t]
            # if bat>0:
            #     print(bat)
        netD_t=D_t-w_t
        if netD_t<0:
            # charging=min(-netD_t,barB-bat)
            netw_t =w_t- D_t
            i=0
            netD_t=0
        else:
            netw_t = 0
            # discharging=min(netD_t,bat)
            # netD_t=netD_t-discharging
            i = netD_t
            emiunit=plant_emi_curve[netD_t]
        if pred[t]<load[t]:
            need = load[t]-pred[t]-netw_t
            if need > 0:
                if  discharging+need<=bat:
                    discharging=discharging+need
                else:
                    i=i+need-(bat-discharging)
                    emiunit += marginal_cost[i]*(need-(bat-discharging))
                    discharging=bat
            else:
                charging =min(barB-bat, netw_t - (load[t]-pred[t]))
        else:
            delta = min(discharging+barB-bat-netw_t,pred[t]-load[t])
            if delta<pred[t]-load[t]:
                emiunit += marginal_cost[i]*(pred[t]-load[t]-delta)
            charging=min(discharging+barB-bat,netw_t+pred[t]-load[t])
        if (i+wind[t]-charging+discharging< load[t]):
            import ipdb;ipdb.set_trace()
        assert i+wind[t]-charging+discharging>=load[t]
        bat=bat+charging-discharging
        decision=np.append(decision,charging-discharging)
        emi_array=np.append(emi_array,emiunit)
        # print(wind[t],load[t],pred[t],bat, charging-discharging, emiunit)
        emitot=emitot+emiunit
        # print(emitot)

    print("minimum emission:", emitot)
    # print(record_emi_array)
    # print("corresponding final battery: ", bat)
    return emitot, emi_array, decision


@njit
def battery_opt_net(net_pred,load,wind,start,end,capacity, pe, barB, online=False):
    emi_min=1e20
    temptot=0
    plant_emi_curve=np.array([0], dtype=np.float64)
    marginal_cost = np.array([0], dtype=np.float64)
    maxcapacity = 0
    for i in range(0,capacity.shape[0]):
        maxcapacity += capacity[i]
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            marginal_cost = np.append(marginal_cost,pe[i])
    net_load = load - wind

    emi_bat_t1 =np.zeros(barB+1)
    emi_bat_t0 = np.zeros(barB+1)
    record_decision = np.zeros((end-start,barB+1))
    for t in range(end,start,-1):
        netD_t = net_pred[t-1]
        netl_t = net_load[t-1]
        if t % 500 == 0:
            print(t)
        for bat_temp in range(barB+1):
            emi_bat_t1[bat_temp] = emi_bat_t0[bat_temp]
            emi_bat_t0[bat_temp] = emi_min

        for bat_last in range(barB+1):
            for bat_cur in range(barB+1):
                ifcharge = bat_cur - bat_last # the amount to charge
                bat = bat_last
                emi_temp = 0
                p2 = netl_t + ifcharge 
                if online == False:
                    if max(p2,0) > netD_t - bat:
                        p1 = max(p2,0)
                    else:
                        p1 = netD_t - bat
                else:
                    p1 = max(0,netD_t-bat)

                emi_temp = plant_emi_curve[p1]
                # i = p1
                emi_temp += marginal_cost[max(p1,p2)]*abs(p1-p2)

                if emi_temp < 0:
                    raise ValueError
                if emi_temp + emi_bat_t1[bat_cur] < emi_bat_t0[bat_last]:
                    emi_bat_t0[bat_last] = emi_temp + emi_bat_t1[bat_cur]
                    record_decision[t-start-1][bat_last] = bat_cur


    decision = np.zeros(end-start)
    bat = round(0)
    for t in range(start,end):
        nxt = record_decision[t-start][bat]
        decision[t-start] = nxt - bat
        bat = round(nxt)
    # import ipdb;ipdb.set_trace()
    emi_tot, emi_array = check_emi_decision_1(plant_emi_curve, marginal_cost, wind, net_pred+wind[:end], load, barB, decision, start, end, online)

    print(emi_tot, emi_bat_t0[0])
    assert abs(emi_tot - emi_bat_t0[0])/emi_tot < 1e-12
    return emi_bat_t0[0], emi_array, decision


def battery_cost_cmh(pred,load,wind,start,end,capacity,pe,rho,barB, theoretical_best = True):
    hatB=math.floor((1-rho)*barB)
    # hatB = barB
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
    power_plant = np.array([0])
    for i in range(0,capacity.shape[0]):
        M= max(M,mc[i])
        m=min(m,mc[i])
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            tempcost=tempcost+mc[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            plant_cost_curve=np.append(plant_cost_curve,tempcost)
            marginal_cost=np.append(marginal_cost,mc[i])
        power_plant=np.append(power_plant, power_plant[-1] + capacity[i])
            #print(plant_emi_curve)

    maxcapcity=len(plant_emi_curve)-1

    if theoretical_best:
        '''
        minghua chen
        '''
        '''
        btheta=(math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2
        print("best theta:", (math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2)
        '''
        theta_mean = (math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2
        cost_tot, cap_tot = 0,0
        for temp_id in range(0,capacity.shape[0]):
            cost_tot += mc[temp_id] * capacity[temp_id]
            cap_tot += capacity[temp_id]
            if cost_tot/cap_tot > theta_mean:
                #with temp_id power plant, average cost is not below thetamean
                #the last one is temp_id - 1 power plant
                print(temp_id, cost_tot/cap_tot, theta_mean)
                if temp_id == 0:
                    btheta = 1e-8
                else:
                    btheta = mc[temp_id - 1] + 1e-8
                break
        theta_list = [btheta]
        id_theta_theo = 0
        while mc[id_theta_theo]<btheta:
            id_theta_theo += 1
    else:
        theta_list = np.concatenate(([0],mc))+1e-8
        # theta_list = [1e-8]
    record_bat=0
    #for theta in np.arange(0,50,0.2):
    for id_theta,theta in enumerate(theta_list):
        # print(theta,theta_list)
        if theoretical_best:
            id_theta = id_theta_theo
        emitot=0
        bat=0

        emi_array=np.array([])
        decision=np.array([])
        for t in range(start,end,1):

            emiunit=0
            grid=0
            charging=0
            discharging=0
            discharging_delta=0
            w_t=wind[t]
            D_t=pred[t]

            i=0
            #different from trivial, we consider the cost to abandon the power, so when wind is not enough, we first let plants to produce power, not let battery discharges.
            netD_t=D_t-w_t
            if w_t>=D_t:
                charging=min(w_t-D_t,barB-bat)
                # w_t = w_t-charging
                netw_t = w_t - D_t - charging
            else:
                # w_t=0
                netw_t = 0
            netD_t=max(0,netD_t)

            price = marginal_cost[netD_t]
            if netD_t>=maxcapcity:
                print("exceed max capacity")
                print(t,D_t,w_t,netD_t)
                print(maxcapcity)
                raise ValueError
            # if netD_t>0:
                # price=(plant_cost_curve[netD_t+1]-plant_cost_curve[netD_t-1])/2
            # if D_t>maxcapcity:
            #     print("exceed max capacity")
            # price = plant_emi_curve[D_t]/D_t
            '''
            minghua chen's mechanism
            '''

            if price<=theta:
                i=netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
            else:
                discharging=min(netD_t,bat)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                emiunit=plant_emi_curve[netD_t-discharging]
            
            netcharge = charging - discharging

            if pred[t]<load[t]:
                need = load[t]-pred[t]-netw_t
                if need > 0:
                    if need <= bat + netcharge:
                        netcharge -= need 
                    else:
                        i=i+need-(bat+netcharge)
                        emiunit += marginal_cost[i]*(need-(bat+netcharge))
                        netcharge = -bat
                else:
                    if netcharge+bat<barB or netcharge+bat>barB:
                        raise ValueError
                    delta = min(-need, barB-bat-netcharge)
                    if -need > barB-bat-netcharge:
                        emiunit += marginal_cost[i]*(-need - (barB-bat-netcharge))
                        if marginal_cost[i]*(-need - (barB-bat-netcharge)) != 0:
                            raise ValueError
                    netcharge += delta
            else:
                delta = min(barB-netcharge-bat-netw_t,pred[t]-load[t])
                if delta<pred[t]-load[t]:
                    emiunit += marginal_cost[i]*(pred[t]-load[t]-delta)
                netcharge += min(barB-netcharge-bat,pred[t]-load[t]+netw_t)

            assert i+wind[t]-netcharge>=load[t]

            if netcharge<-bat or bat+netcharge >barB:
                print(t,charging,discharging)
                import ipdb;ipdb.set_trace()
                raise ValueError

            bat=bat+netcharge

            decision=np.append(decision,netcharge)
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
    print("corresponding theta:", record_theta)
    # # print(record_emi_array)
    # print("corresponding final battery: ", record_bat)
    return emi_min,record_theta,record_emi_array,record_decision

def battery_cost_min_hatB(pred,load,wind,start,end,capacity,pe,rho,barB, theoretical_best):
    hatB=math.floor((1-rho)*barB)
    # hatB = barB
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
    power_plant = np.array([0])
    for i in range(0,capacity.shape[0]):
        M= max(M,mc[i])
        m=min(m,mc[i])
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            tempcost=tempcost+mc[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            plant_cost_curve=np.append(plant_cost_curve,tempcost)
            marginal_cost=np.append(marginal_cost,mc[i])
        power_plant=np.append(power_plant, power_plant[-1] + capacity[i])
            #print(plant_emi_curve)

    maxcapcity=len(plant_emi_curve)-1

    # print(M)
    # print(m)
    # for i in range(1,maxcapcity+1):
    #     M = max(M, plant_emi_curve[i]/i)
    #     m = min(m, plant_emi_curve[i]/i)
    if theoretical_best:
        '''
        minghua chen
        '''
        '''
        btheta=(math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2
        print("best theta:", (math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2)
        '''
        theta_mean = math.sqrt(M*m)
        cost_tot, cap_tot = 0,0
        for temp_id in range(0,capacity.shape[0]):
            cost_tot += mc[temp_id] * capacity[temp_id]
            cap_tot += capacity[temp_id]
            if cost_tot/cap_tot > theta_mean:
                #with temp_id power plant, average cost is not below thetamean
                #the last one is temp_id - 1 power plant
                print(temp_id, cost_tot/cap_tot, theta_mean)
                if temp_id == 0:
                    btheta = 1e-8
                else:
                    btheta = mc[temp_id - 1] + 1e-8
                break
        theta_list = [btheta]
        id_theta_theo = 0
        while mc[id_theta_theo]<btheta:
            id_theta_theo += 1
    else:
        theta_list = np.concatenate(([0],mc))+1e-8
        # theta_list = [1e-8]
    record_bat=0
    #for theta in np.arange(0,50,0.2):
    for id_theta,theta in enumerate(theta_list):
        # print(theta,theta_list)
        if theoretical_best:
            id_theta = id_theta_theo
        emitot=0
        bat=0

        emi_array=np.array([])
        decision=np.array([])
        for t in range(start,end,1):

            emiunit=0
            grid=0
            charging=0
            discharging=0
            discharging_delta=0
            w_t=wind[t]
            D_t=pred[t]

            i=0
            #different from trivial, we consider the cost to abandon the power, so when wind is not enough, we first let plants to produce power, not let battery discharges.
            netD_t=D_t-w_t
            if w_t>=D_t:
                charging=min(w_t-D_t,barB-bat)
                # w_t = w_t-charging
                netw_t = w_t - D_t - charging
            else:
                # w_t=0
                netw_t = 0
            netD_t=max(0,netD_t)

            price = marginal_cost[netD_t]
            if netD_t>=maxcapcity:
                print("exceed max capacity")
                print(t,D_t,w_t,netD_t)
                print(maxcapcity)
                raise ValueError
            # if netD_t>0:
                # price=(plant_cost_curve[netD_t+1]-plant_cost_curve[netD_t-1])/2
            # if D_t>maxcapcity:
            #     print("exceed max capacity")
            # price = plant_emi_curve[D_t]/D_t
            '''
            minghua chen's mechanism
            '''
            ''' 
            if price<=theta:
                i=netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
            else:
                discharging=min(netD_t,bat)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                emiunit=plant_emi_curve[netD_t-discharging]
            '''

            '''
            non-convex mechanism
            '''
            if price<=theta:
                i=netD_t+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))
            else:
                discharging=min(netD_t-power_plant[id_theta],bat)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                    raise ValueError
                emiunit=plant_emi_curve[netD_t-discharging]
            
            netcharge = charging - discharging

            if pred[t]<load[t]:
                need = load[t]-pred[t]-netw_t
                if need > 0:
                    # if  discharging+need<=bat:
                    if need <= bat + netcharge:
                        netcharge -= need 
                        # discharging=discharging+need
                    else:
                        i=i+need-(bat+netcharge)
                        emiunit += marginal_cost[i]*(need-(bat+netcharge))
                        netcharge = -bat
                else:
                    if netcharge+bat<barB or netcharge+bat>barB:
                        raise ValueError
                    delta = min(-need, barB-bat-netcharge)
                    if -need > barB-bat-netcharge:
                        emiunit += marginal_cost[i]*(-need - (barB-bat-netcharge))
                        if marginal_cost[i]*(-need - (barB-bat-netcharge)) != 0:
                            raise ValueError
                    netcharge += delta
            else:
                delta = min(barB-netcharge-bat-netw_t,pred[t]-load[t])
                if delta<pred[t]-load[t]:
                    emiunit += marginal_cost[i]*(pred[t]-load[t]-delta)
                netcharge += min(barB-netcharge-bat,pred[t]-load[t]+netw_t)

            assert i+wind[t]-netcharge>=load[t]

            if netcharge<-bat or bat+netcharge >barB:
                print(t,charging,discharging)
                import ipdb;ipdb.set_trace()
                raise ValueError

            bat=bat+netcharge

            # if charging>barB or discharging>barB:
            #     print(t,charging,discharging)
            #     raise ValueError

            decision=np.append(decision,netcharge)
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
    print("corresponding theta:", record_theta)
    # # print(record_emi_array)
    # print("corresponding final battery: ", record_bat)
    return emi_min,record_theta,record_emi_array,record_decision



def battery_cost_cmh_trivialfirst(pred,load,wind,start,end,capacity,pe,rho,barB, theoretical_best = True):
    hatB=math.floor((1-rho)*barB)
    # hatB = barB
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
    power_plant = np.array([0])
    for i in range(0,capacity.shape[0]):
        M= max(M,mc[i])
        m=min(m,mc[i])
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            tempcost=tempcost+mc[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            plant_cost_curve=np.append(plant_cost_curve,tempcost)
            marginal_cost=np.append(marginal_cost,mc[i])
        power_plant=np.append(power_plant, power_plant[-1] + capacity[i])
            #print(plant_emi_curve)

    maxcapcity=len(plant_emi_curve)-1

    if theoretical_best:
        '''
        minghua chen
        '''
        '''
        btheta=(math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2
        print("best theta:", (math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2)
        '''
        theta_mean = (math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2
        cost_tot, cap_tot = 0,0
        for temp_id in range(0,capacity.shape[0]):
            cost_tot += mc[temp_id] * capacity[temp_id]
            cap_tot += capacity[temp_id]
            if cost_tot/cap_tot > theta_mean:
                #with temp_id power plant, average cost is not below thetamean
                #the last one is temp_id - 1 power plant
                print(temp_id, cost_tot/cap_tot, theta_mean)
                if temp_id == 0:
                    btheta = 1e-8
                else:
                    btheta = mc[temp_id - 1] + 1e-8
                break
        theta_list = [btheta]
        id_theta_theo = 0
        while mc[id_theta_theo]<btheta:
            id_theta_theo += 1
    else:
        theta_list = np.concatenate(([0],mc))+1e-8
        # theta_list = [1e-8]
    record_bat=0
    #for theta in np.arange(0,50,0.2):
    for id_theta,theta in enumerate(theta_list):
        # print(theta,theta_list)
        if theoretical_best:
            id_theta = id_theta_theo
        emitot=0
        bat=0

        emi_array=np.array([])
        decision=np.array([])
        for t in range(start,end,1):

            emiunit=0
            grid=0
            charging=0
            discharging=0
            discharging_delta = 0
            w_t=wind[t]
            D_t=pred[t]

            i=0
            #different from trivial, we consider the cost to abandon the power, so when wind is not enough, we first let plants to produce power, not let battery discharges.
            netD_t=D_t-w_t
            if w_t>=D_t:
                charging=min(w_t-D_t,barB-bat)
                # w_t = w_t-charging
                netw_t = w_t - D_t - charging
            else:
                # w_t=0
                netw_t = 0
                discharging_delta = min(netD_t,bat)
                netD_t=netD_t-discharging_delta
                bat = bat - discharging_delta
            netD_t=max(0,netD_t)

            price = marginal_cost[netD_t]
            if netD_t>=maxcapcity:
                print("exceed max capacity")
                print(t,D_t,w_t,netD_t)
                print(maxcapcity)
                raise ValueError
            # if netD_t>0:
                # price=(plant_cost_curve[netD_t+1]-plant_cost_curve[netD_t-1])/2
            # if D_t>maxcapcity:
            #     print("exceed max capacity")
            # price = plant_emi_curve[D_t]/D_t
            '''
            minghua chen's mechanism
            '''

            if price<=theta:
                i=netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
            else:
                discharging+=min(netD_t,bat)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                emiunit=plant_emi_curve[netD_t-discharging]
            
            netcharge = charging - discharging

            if pred[t]<load[t]:
                need = load[t]-pred[t]-netw_t
                if need > 0:
                    if need <= bat + netcharge:
                        netcharge -= need 
                    else:
                        i=i+need-(bat+netcharge)
                        emiunit += marginal_cost[i]*(need-(bat+netcharge))
                        netcharge = -bat
                else:
                    if netcharge+bat<barB or netcharge+bat>barB:
                        raise ValueError
                    delta = min(-need, barB-bat-netcharge)
                    if -need > barB-bat-netcharge:
                        emiunit += marginal_cost[i]*(-need - (barB-bat-netcharge))
                        if marginal_cost[i]*(-need - (barB-bat-netcharge)) != 0:
                            raise ValueError
                    netcharge += delta
            else:
                delta = min(barB-netcharge-bat-netw_t,pred[t]-load[t])
                if delta<pred[t]-load[t]:
                    emiunit += marginal_cost[i]*(pred[t]-load[t]-delta)
                netcharge += min(barB-netcharge-bat,pred[t]-load[t]+netw_t)

            assert i+wind[t]-netcharge+discharging_delta>=load[t]

            if netcharge<-bat or bat+netcharge >barB:
                print(t,charging,discharging)
                import ipdb;ipdb.set_trace()
                raise ValueError

            bat=bat+netcharge

            decision=np.append(decision,netcharge)
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
    print("corresponding theta:", record_theta)
    # # print(record_emi_array)
    # print("corresponding final battery: ", record_bat)
    return emi_min,record_theta,record_emi_array,record_decision


def battery_cost_min_trivialfirst(pred,load,wind,start,end,capacity,pe,rho,barB, theoretical_best):
    # hatB=math.floor((1-rho)*barB)
    hatB = barB
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
    power_plant = np.array([0])
    for i in range(0,capacity.shape[0]):
        M= max(M,mc[i])
        m=min(m,mc[i])
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            tempcost=tempcost+mc[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            plant_cost_curve=np.append(plant_cost_curve,tempcost)
            marginal_cost=np.append(marginal_cost,mc[i])
        power_plant=np.append(power_plant, power_plant[-1] + capacity[i])
            #print(plant_emi_curve)

    maxcapcity=len(plant_emi_curve)-1

    # print(M)
    # print(m)
    # for i in range(1,maxcapcity+1):
    #     M = max(M, plant_emi_curve[i]/i)
    #     m = min(m, plant_emi_curve[i]/i)
    if theoretical_best:
        '''
        minghua chen
        '''
        '''
        btheta=(math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2
        print("best theta:", (math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2)
        '''
        theta_mean = math.sqrt(M*m)
        cost_tot, cap_tot = 0,0
        for temp_id in range(0,capacity.shape[0]):
            cost_tot += mc[temp_id] * capacity[temp_id]
            cap_tot += capacity[temp_id]
            if cost_tot/cap_tot > theta_mean:
                #with temp_id power plant, average cost is not below thetamean
                #the last one is temp_id - 1 power plant
                print(temp_id, cost_tot/cap_tot, theta_mean)
                if temp_id == 0:
                    btheta = 1e-8
                else:
                    btheta = mc[temp_id - 1] + 1e-8
                break
        theta_list = [btheta]
        id_theta_theo = 0
        while mc[id_theta_theo]<btheta:
            id_theta_theo += 1
    else:
        theta_list = np.concatenate(([0],mc))+1e-8
        # theta_list = [1e-8]
    record_bat=0
    #for theta in np.arange(0,50,0.2):
    for id_theta,theta in enumerate(theta_list):
        # print(theta,theta_list)
        if theoretical_best:
            id_theta = id_theta_theo
        emitot=0
        bat=0

        emi_array=np.array([])
        decision=np.array([])
        for t in range(start,end,1):

            emiunit=0
            grid=0
            charging=0
            discharging=0
            discharging_delta=0
            w_t=wind[t]
            D_t=pred[t]

            i=0
            #different from trivial, we consider the cost to abandon the power, so when wind is not enough, we first let plants to produce power, not let battery discharges.
            netD_t=D_t-w_t
            if w_t>=D_t:
                charging=min(w_t-D_t,barB-bat)
                # w_t = w_t-charging
                netw_t = w_t - D_t - charging
            else:
                # w_t=0
                netw_t = 0
                discharging_delta = min(netD_t,bat)
                netD_t=netD_t-discharging_delta
                bat = bat - discharging_delta
            netD_t=max(0,netD_t)

            price = marginal_cost[netD_t]
            if netD_t>=maxcapcity:
                print("exceed max capacity")
                print(t,D_t,w_t,netD_t)
                print(maxcapcity)
                raise ValueError
            # if netD_t>0:
                # price=(plant_cost_curve[netD_t+1]-plant_cost_curve[netD_t-1])/2
            # if D_t>maxcapcity:
            #     print("exceed max capacity")
            # price = plant_emi_curve[D_t]/D_t
            '''
            minghua chen's mechanism
            '''
            ''' 
            if price<=theta:
                i=netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
            else:
                discharging=min(netD_t,bat)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                emiunit=plant_emi_curve[netD_t-discharging]
            '''

            '''
            non-convex mechanism
            '''
            if price<=theta:
                i=netD_t+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))
            else:
                discharging=min(netD_t-power_plant[id_theta],bat)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                    raise ValueError
                emiunit=plant_emi_curve[netD_t-discharging]
            
            netcharge = charging - discharging

            if pred[t]<load[t]:
                need = load[t]-pred[t]-netw_t
                if need > 0:
                    # if  discharging+need<=bat:
                    if need <= bat + netcharge:
                        netcharge -= need 
                        # discharging=discharging+need
                    else:
                        i=i+need-(bat+netcharge)
                        emiunit += marginal_cost[i]*(need-(bat+netcharge))
                        netcharge = -bat
                else:
                    if netcharge+bat<barB or netcharge+bat>barB:
                        raise ValueError
                    delta = min(-need, barB-bat-netcharge)
                    if -need > barB-bat-netcharge:
                        emiunit += marginal_cost[i]*(-need - (barB-bat-netcharge))
                        if marginal_cost[i]*(-need - (barB-bat-netcharge)) != 0:
                            raise ValueError
                    netcharge += delta
            else:
                delta = min(barB-netcharge-bat-netw_t,pred[t]-load[t])
                if delta<pred[t]-load[t]:
                    emiunit += marginal_cost[i]*(pred[t]-load[t]-delta)
                netcharge += min(barB-netcharge-bat,pred[t]-load[t]+netw_t)

            assert i+wind[t]-netcharge+discharging_delta>=load[t]

            if netcharge<-bat or bat+netcharge >barB:
                print(t,charging,discharging)
                import ipdb;ipdb.set_trace()
                raise ValueError

            bat=bat+netcharge

            # if charging>barB or discharging>barB:
            #     print(t,charging,discharging)
            #     raise ValueError

            decision=np.append(decision,netcharge)
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
    print("corresponding theta:", record_theta)
    # # print(record_emi_array)
    # print("corresponding final battery: ", record_bat)
    return emi_min,record_theta,record_emi_array,record_decision


def battery_cost_min_hatB_trivialfirst(pred,load,wind,start,end,capacity,pe,rho,barB, theoretical_best):
    hatB=math.floor((1-rho)*barB)
    # hatB = barB
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
    power_plant = np.array([0])
    for i in range(0,capacity.shape[0]):
        M= max(M,mc[i])
        m=min(m,mc[i])
        for j in range(0,capacity[i]):
            temptot=temptot+pe[i]
            tempcost=tempcost+mc[i]
            plant_emi_curve=np.append(plant_emi_curve,temptot)
            plant_cost_curve=np.append(plant_cost_curve,tempcost)
            marginal_cost=np.append(marginal_cost,mc[i])
        power_plant=np.append(power_plant, power_plant[-1] + capacity[i])
            #print(plant_emi_curve)

    maxcapcity=len(plant_emi_curve)-1

    # print(M)
    # print(m)
    # for i in range(1,maxcapcity+1):
    #     M = max(M, plant_emi_curve[i]/i)
    #     m = min(m, plant_emi_curve[i]/i)
    if theoretical_best:
        '''
        minghua chen
        '''
        '''
        btheta=(math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2
        print("best theta:", (math.sqrt(rho*rho*(M-m)*(M-m)+4*M*m)-rho*(M-m))/2)
        '''
        theta_mean = math.sqrt(M*m)
        cost_tot, cap_tot = 0,0
        for temp_id in range(0,capacity.shape[0]):
            cost_tot += mc[temp_id] * capacity[temp_id]
            cap_tot += capacity[temp_id]
            if cost_tot/cap_tot > theta_mean:
                #with temp_id power plant, average cost is not below thetamean
                #the last one is temp_id - 1 power plant
                print(temp_id, cost_tot/cap_tot, theta_mean)
                if temp_id == 0:
                    btheta = 1e-8
                else:
                    btheta = mc[temp_id - 1] + 1e-8
                break
        theta_list = [btheta]
        id_theta_theo = 0
        while mc[id_theta_theo]<btheta:
            id_theta_theo += 1
    else:
        theta_list = np.concatenate(([0],mc))+1e-8
        # theta_list = [1e-8]
    record_bat=0
    #for theta in np.arange(0,50,0.2):
    for id_theta,theta in enumerate(theta_list):
        # print(theta,theta_list)
        if theoretical_best:
            id_theta = id_theta_theo
        emitot=0
        bat=0

        emi_array=np.array([])
        decision=np.array([])
        for t in range(start,end,1):

            emiunit=0
            grid=0
            charging=0
            discharging=0
            discharging_delta=0
            w_t=wind[t]
            D_t=pred[t]

            i=0
            #different from trivial, we consider the cost to abandon the power, so when wind is not enough, we first let plants to produce power, not let battery discharges.
            netD_t=D_t-w_t
            if w_t>=D_t:
                charging=min(w_t-D_t,barB-bat)
                # w_t = w_t-charging
                netw_t = w_t - D_t - charging
            else:
                # w_t=0
                netw_t = 0
                discharging_delta = min(netD_t,bat)
                netD_t=netD_t-discharging_delta
                bat = bat - discharging_delta
            netD_t=max(0,netD_t)

            price = marginal_cost[netD_t]
            if netD_t>=maxcapcity:
                print("exceed max capacity")
                print(t,D_t,w_t,netD_t)
                print(maxcapcity)
                raise ValueError
            # if netD_t>0:
                # price=(plant_cost_curve[netD_t+1]-plant_cost_curve[netD_t-1])/2
            # if D_t>maxcapcity:
            #     print("exceed max capacity")
            # price = plant_emi_curve[D_t]/D_t
            '''
            minghua chen's mechanism
            '''
            ''' 
            if price<=theta:
                i=netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(maxcapcity-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(maxcapcity-netD_t,max(hatB-bat-charging,0))
            else:
                discharging=min(netD_t,bat)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                emiunit=plant_emi_curve[netD_t-discharging]
            '''

            '''
            non-convex mechanism
            '''
            if price<=theta:
                i=netD_t+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))
                emiunit=plant_emi_curve[netD_t+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))]
                charging=charging+min(power_plant[id_theta]-netD_t,max(hatB-bat-charging,0))
            else:
                discharging=min(netD_t-power_plant[id_theta],bat)
                i=netD_t-discharging
                if i>maxcapcity:
                    print(t,netD_t,D_t,w_t,discharging)
                    print(maxcapcity)
                    raise ValueError
                emiunit=plant_emi_curve[netD_t-discharging]
            
            netcharge = charging - discharging

            if pred[t]<load[t]:
                need = load[t]-pred[t]-netw_t
                if need > 0:
                    # if  discharging+need<=bat:
                    if need <= bat + netcharge:
                        netcharge -= need 
                        # discharging=discharging+need
                    else:
                        i=i+need-(bat+netcharge)
                        emiunit += marginal_cost[i]*(need-(bat+netcharge))
                        netcharge = -bat
                else:
                    if netcharge+bat<barB or netcharge+bat>barB:
                        raise ValueError
                    delta = min(-need, barB-bat-netcharge)
                    if -need > barB-bat-netcharge:
                        emiunit += marginal_cost[i]*(-need - (barB-bat-netcharge))
                        if marginal_cost[i]*(-need - (barB-bat-netcharge)) != 0:
                            raise ValueError
                    netcharge += delta
            else:
                delta = min(barB-netcharge-bat-netw_t,pred[t]-load[t])
                if delta<pred[t]-load[t]:
                    emiunit += marginal_cost[i]*(pred[t]-load[t]-delta)
                netcharge += min(barB-netcharge-bat,pred[t]-load[t]+netw_t)

            assert i+wind[t]-netcharge+discharging_delta>=load[t]

            if netcharge<-bat or bat+netcharge >barB:
                print(t,charging,discharging)
                import ipdb;ipdb.set_trace()
                raise ValueError

            bat=bat+netcharge

            # if charging>barB or discharging>barB:
            #     print(t,charging,discharging)
            #     raise ValueError

            decision=np.append(decision,netcharge)
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
    print("corresponding theta:", record_theta)
    # # print(record_emi_array)
    # print("corresponding final battery: ", record_bat)
    return emi_min,record_theta,record_emi_array,record_decision



    

