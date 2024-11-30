import numpy as np
import pandas
import scipy
from scipy.io import loadmat
import math

import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
from battery_mechanisms_2023 import battery_trivial,battery_cost_min,battery_opt_1,battery_opt_net, battery_cost_integral
from battery_mechanisms_2023 import battery_trivial_secondphase, battery_cost_cmh,battery_cost_min_hatB,battery_cost_cmh_trivialfirst, battery_cost_min_trivialfirst, battery_cost_min_hatB_trivialfirst

import math
import pandas as pd
import os
from datasets.data import *
from hyperparameters import loadratio, windratio,solar_ratio

n=151
start=6024
length=2728
# length = 10
end=start+length
# loadratio=1.4
# mixed_pred_pr=0.2
import argparse

parser = argparse.ArgumentParser(description='aims')
parser.add_argument('-trainaim', type=str, help='pred or cost or cost_strg or cost_strg_strategy')
parser.add_argument('-net', type=bool, default = False)
parser.add_argument('-mixed', type=float, help="proportion of attacked data", default = 0)
parser.add_argument('-prepath', type=str)

args = parser.parse_args()
adv_example = args.trainaim#"pred" or "cost"
adv_attack = args.trainaim
mixed = args.mixed
prepath = args.prepath
if args.net == True:
    data_file = "adv_" + adv_example + "_attack_" + adv_attack + "_net"
else:
    data_file = "adv_" + adv_example + "_attack_" + adv_attack
if prepath == "test_with_plain_model_load_attack_bound":
    attacked = "attacked"
    bound_list = [1,2,3,4,5,6,7,8,9,10]
else:
    attacked = ""
    bound_list = [0,1,2,3,4,5,6,7,8,9,10]
bound_list = [3]
print(data_file)

# test_type = "" #"mixed" or ""
# data_file = "adv_cost_attack_cost"
# data_file = "adv_pred_attack_pred"
# data_file = "adv_cost_attack_cost_net"
# data_file = "adv_cost_strg_attack_cost_strg_net"
# data_file = "adv_cost_strg_strategy_attack_cost_strg_strategy_net"
# data_file = "adv_pred_attack_pred_net"
# data_file = "adv_results_debug_test"
# data_file = "results_2022"
# data_file = "adv_cost_attack_cost_debug_test" 
windratio_array=[windratio]
# bound_list = [6]
# print(data_file)

m=loadmat("QuantileSolutionRampStatic161028.mat")
path = os.path.join(os.path.abspath('.'), 'datasets/newdata.csv')  # Directory for dataset
df = load_dataset(path=path)
# netloadnw=np.array(m['netloadnw']).reshape(-1)
# import ipdb;ipdb.set_trace()
# netloadnw.to_csv('year_demand.csv')
# dataframe=pandas.DataFrame(netloadnw)
# dataframe.to_csv('year_demand.csv')
# print(netloadnw[start:end])
# print(netloadnw[start:end].shape)

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
capacity[-1]+=100000
pe = mc
emi_type = 'mc'
def mixed_pred(pred, load, start, end, mixed_pred_pr):
    length = end - start
    rn = np.random.binomial(1,mixed_pred_pr, length)
    print(np.sum(rn==1),np.sum(rn==0))
    return np.concatenate((pred[:start],pred[start:end]*rn + load[start:end]*(1-rn)))

def run(bound):
    record_improvement_battery=[]
    record_improvement_trivial=[]
    record_improvement_battery_rho=[]
    record_improvement_battery_cost=[]
    record_improvement_battery_cost_min=[]
    record_improvement_battery_cost_mintheta=[]
    record_improvement_opt=[]

    record_emi_no=[]
    record_emi_opt=[]
    record_emi_online_opt=[]
    record_emi_battery_cost_min=[]
    record_emi_battery_cost_mintheta=[]
    record_emi_trivial=[]
    record_emi_mixed=[]

    record_improvement_battery_pc=[]
    record_improvement_trivial_pc=[]
    record_improvement_opt_pc=[]
    record_improvement_battery_rho_pc=[]
    record_improvement_battery_cost_pc=[]
    record_improvement_battery_cost_min_pc=[]
    record_improvement_battery_cost_mintheta_pc=[]


    for windratio in windratio_array:
    #windratio=6.5

        # load=np.array(loadratio*m['netloadnw']).reshape(-1)
        wind=np.array(m['wexp']).reshape(-1)
        load = np.array(loadratio*df['actual'][:wind.shape[0]]).reshape(-1)
        solar=np.load("changedsolar.npy")*solar_ratio#16000
        wind=windratio*(wind+solar)

        load=np.rint(load)
        # print(load)
        load=load.astype(np.int64)
        # print(load)
        wind=np.rint(wind).astype(np.int64)
        # if bound <-0.001:
        #     b = load / loadratio
        # elif bound <0.001:
        #     b = np.genfromtxt(f'{data_file}/pred_data.csv',dtype=float,delimiter=',')
        #     b= b[:,0].reshape(-1)
        #     zero=np.zeros(start).reshape(-1)
        #     b=np.append(zero,b)
        #     print(b.shape)
        # else:
            
        print("load_attack_bound%.2f_result"%(bound))
        # b = np.genfromtxt(f'{data_file}/test_with_plain_model_load_attack_bound%.2f_result.csv'%(bound),dtype=float,delimiter=',')
        if bound<-1e-5:
            b = load[start:end].reshape(-1)
        else:
            b = np.genfromtxt(f'{data_file}/{prepath}%.2f_result.csv'%(bound),dtype=float,delimiter=',')
            b= b[:,0].reshape(-1)
        zero=np.zeros(start).reshape(-1)
        b=np.append(zero,b)
        print(b.shape)
        forecast = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.00_result.csv',dtype=float,delimiter=',')
        forecast = forecast[:,0].reshape(-1)
        forecast = np.append(zero, forecast)
        #print(end)

        # if bound<0:
        #     pred = np.copy(load)
        # else:
        # import ipdb;ipdb.set_trace()
        if data_file[-3:] == "net":
            pred = np.rint(b + wind[:end]).astype(np.int64)
            forecast = np.rint(forecast + wind[:end]).astype(np.int64)
        else:
            pred=np.rint(b).astype(np.int64)
        # if data_file =="adv_cost_attack_cost" or data_file =="adv_cost_attack_cost_debug_test":
        #     pred = np.rint(b).astype(np.int64)
        
        for i in range(start,start+5):
            print(load[i],pred[i])
        # import ipdb;ipdb.set_trace()
        # plt.plot(load[start:end],label="load")
        # plt.plot(pred[start:end],label="pred")
        # plt.legend()
        # plt.savefig("temp.jpg")
        # raise ValueError
        r_sum=0
        a_sum=0
        for i in range(start,end):
            if wind[i]>load[i]:
                r_sum=r_sum+wind[i]-load[i]
            else:
                a_sum=a_sum+load[i]-wind[i]
        if mixed >0:
            pred = mixed_pred(pred, forecast, start, end, mixed)
        rho=r_sum/a_sum
        print("="*25,"rho = ",rho,"="*25)
        # print("emi_no",emi_no)
        # raise ValueError
        # if test_type == "mixed":
        #     emi_mixed, theta_mixed, emi_array_mixed, decision_mixed=battery_cost_integral(pred,load,wind,start,end,capacity,pe,rho,16000,a_sum/(end-start))
        #     print("cost mixed", emi_mixed)
        #     record_emi_mixed.append(emi_mixed)
        #     plt.plot(theta_mixed)
        #     plt.xlabel("T")
        #     plt.ylabel("theta")
        #     plt.savefig(f"{data_file}/%.2ftheta_mixed%.2f.jpg"%(bound,mixed_pred_pr))
        #     # print("cost",emi_array_mixed)
        #     np.save(f"{data_file}/%.2f%s%.1fwind_record_theta_mixed%.2f"% (bound, emi_type,windratio,mixed_pred_pr), theta_mixed)
        #     np.save(f"{data_file}/%.2f%s%.1fwind_emi_array_mixed%.2f"% (bound, emi_type,windratio,mixed_pred_pr), emi_mixed)
        #     np.save(f"{data_file}/%.2f%s%.1fwind_decision_mixed%.2f"% (bound, emi_type,windratio,mixed_pred_pr), decision_mixed)
        print("*"*10,'trivial',"*"*10)
        # import ipdb;ipdb.set_trace()
        barB = 16000
        emi_cost_hatB_trivialfirst, emi_cost_hatB_trivialfirst_theta,emi_array_cost_hatB_trivialfirst, decision_cost_hatB_trivialfirst = battery_cost_min_hatB_trivialfirst(pred,load,wind,start,end,capacity,pe,rho,barB, theoretical_best = True)

        [emi_trivial,emi_array_trivial,decision_trivial]=battery_trivial(pred,load,wind,start,end,capacity,pe,16000)
        # # print("cost",emi_array_trivial)
        # # import ipdb; ipdb.set_trace()
        [emi_no,emi_array_no,decision_no]=battery_trivial(pred,load,wind,start,end,capacity,pe,0)

        # print("*"*10,'battery to minimize cost by osc',"*"*10)
        # [emi_battery_cost_min,theta_battery_cost_min,emi_array_battery_cost_min,decision_battery_cost_min]=battery_cost_min(pred,load,wind,start,end,capacity,pe,rho,16000, theoretical_best = True)
        # print('best theta%.1f='% (windratio),theta_battery_cost_min)
        # print("*"*10,'battery to minimize cost from best theta',"*"*10)
        # [emi_battery_cost_mintheta,theta_battery_cost_mintheta,emi_array_battery_cost_mintheta,decision_battery_cost_mintheta]=battery_cost_min(pred,load,wind,start,end,capacity,pe,rho,16000, theoretical_best = False)
        # print("*"*10,'no battery',"*"*10)
        # # [emi_no,theta_no,emi_array_no,decision_no]=battery_simple(pred,wind,start,end,capacity,pe,0.5,0.51,0.1,0)
        # # print("emi_no",emi_no)
        # if data_file[-3:]=="net":
        #     emi_opt, emi_array_opt, decision_opt=battery_opt_net(pred-wind[:end],load,wind,start,end,capacity,pe,16000)
        #     emi_online_opt, emi_array_online_opt, decision_online_opt=battery_opt_net(pred-wind[:end],load,wind,start,end,capacity,pe,16000,True)
        #     # print("emi opt emi online opt",emi_opt,emi_online_opt)
        # else:
        #     emi_opt, emi_array_opt, decision_opt=battery_opt_1(pred,load,wind,start,end,capacity,pe,16000,False)
        #     emi_online_opt, emi_array_online_opt, decision_online_opt=battery_opt_1(pred,load,wind,start,end,capacity,pe,16000,True)



        # emi_trivial_secondphase = battery_trivial_secondphase(pred,load,wind,start,end,capacity,pe,16000)[0]
        # emi_cost_cmh = battery_cost_cmh(pred,load,wind,start,end,capacity,pe,rho,16000)[0]
        # emi_cost_cmh_trivialfirst = battery_cost_cmh_trivialfirst(pred,load,wind,start,end,capacity,pe,rho,16000)[0]
        # emi_cost_min_hatB = battery_cost_min_hatB(pred,load,wind,start,end,capacity,pe,rho,16000, theoretical_best = False)[0]
        # emi_cost_min_trivialfirst = battery_cost_min_trivialfirst(pred,load,wind,start,end,capacity,pe,rho,16000, theoretical_best = False)[0]
        emi_cost_min_hatB_trivialfirst, emi_cost_min_hatB_trivialfirst_theta,emi_array_cost_min_hatB_trivialfirst, decision_cost_min_hatB_trivialfirst = battery_cost_min_hatB_trivialfirst(pred,load,wind,start,end,capacity,pe,rho,16000, theoretical_best = False)

        # print("battery_trivial_secondphase", emi_trivial_secondphase)
        # print("battery_cost_cmh", emi_cost_cmh)
        # print("battery_cost_cmh_trivialfirst", emi_cost_cmh_trivialfirst)
        
        # print("battery_cost_min_hatB", emi_cost_min_hatB)
        # print("battery_cost_min_trivialfirst", emi_cost_min_trivialfirst)
        # print("battery_cost_min_hatB_trivialfirst", emi_cost_min_hatB_trivialfirst)
        # print("emi opt emi online opt",emi_opt,emi_online_opt, emi_cost_min_hatB_trivialfirst_theta)

        # emi_opt = 0

        # print('emi_no%.1f='% (windratio),emi_no)
        # print('emi_trivial%.1f='% (windratio),emi_trivial)
        # print('emi_battery_cost_min%.1f='% (windratio),emi_battery_cost_min)
        # print('emi_battery_cost_mintheta%.1f='% (windratio),emi_battery_cost_mintheta)
        # print('emi_battery_cost_optimal%.1f='% (windratio),emi_opt)

        # record_emi_no.append(emi_no)
        # record_emi_battery_cost_min.append(emi_battery_cost_min)
        # record_emi_battery_cost_mintheta.append(emi_battery_cost_mintheta)
        # record_emi_trivial.append(emi_trivial)
        # # record_emi_opt.append(emi_opt)
        # record_emi_online_opt.append(emi_online_opt)

        # print('emi_trivial_improve%.1f='% (windratio),1-emi_trivial/emi_no)
        # print('emi_battery_cost_min_improve%.1f='% (windratio),1-emi_battery_cost_min/emi_no)
        # print('emi_battery_cost_mintheta_improve%.1f='% (windratio),1-emi_battery_cost_mintheta/emi_no)
        # print('emi_battery_cost_optimal_improve%.1f='% (windratio),1 - emi_opt/emi_no)
        
        # record_improvement_battery_cost_min.append(emi_no-emi_battery_cost_min)
        # record_improvement_battery_cost_mintheta.append(emi_no-emi_battery_cost_mintheta)
        # record_improvement_trivial.append(emi_no-emi_trivial)
        # record_improvement_opt.append(emi_no-emi_opt)

        # record_improvement_battery_cost_min_pc.append(1-emi_battery_cost_min/emi_no)
        # record_improvement_battery_cost_mintheta_pc.append(1-emi_battery_cost_mintheta/emi_no)
        # record_improvement_trivial_pc.append(1-emi_trivial/emi_no)
        # record_improvement_opt_pc.append(1-emi_opt/emi_no)

        # np.save(f"{data_file}/%.2f%s%.1fwind_emi_array_battery_cost_min{attacked}"% (bound, emi_type,windratio), emi_array_battery_cost_min)
        # np.save(f"{data_file}/%.2f%s%.1fwind_emi_array_battery_cost_mintheta{attacked}"% (bound, emi_type,windratio), emi_array_battery_cost_mintheta)
        np.save(f"{data_file}/%.2f%s%.1fwind_emi_array_no{attacked}"% (bound, emi_type,windratio), emi_array_no)
        np.save(f"{data_file}/%.2f%s%.1fwind_emi_array_trivial{attacked}"% (bound, emi_type,windratio), emi_array_trivial)
        # np.save(f"{data_file}/%.2f%s%.1fwind_emi_array_opt{attacked}"% (bound, emi_type,windratio), emi_array_opt)
        # np.save(f"{data_file}/%.2f%s%.1fwind_emi_array_online_opt{attacked}"% (bound, emi_type,windratio), emi_array_online_opt)
        np.save(f"{data_file}/%.2f%s%.1fwind_emi_array_min_hatB_trivialfirst{attacked}"% (bound, emi_type,windratio), emi_array_cost_min_hatB_trivialfirst)
        np.save(f"{data_file}/%.2f%s%.1fwind_emi_array_theory{attacked}"% (bound, emi_type,windratio), emi_array_cost_hatB_trivialfirst)

        # np.save(f"{data_file}/%.2f%s%.1fwind_decision_battery_cost_min{attacked}"% (bound, emi_type,windratio), decision_battery_cost_min)
        # np.save(f"{data_file}/%.2f%s%.1fwind_decision_battery_cost_mintheta{attacked}"% (bound, emi_type,windratio), decision_battery_cost_mintheta)
        np.save(f"{data_file}/%.2f%s%.1fwind_decision_trivial{attacked}"% (bound, emi_type,windratio), decision_trivial)
        # np.save(f"{data_file}/%.2f%s%.1fwind_decision_opt{attacked}"% (bound, emi_type,windratio), decision_opt)
        # np.save(f"{data_file}/%.2f%s%.1fwind_decision_online_opt{attacked}"% (bound, emi_type,windratio), decision_online_opt)
        np.save(f"{data_file}/%.2f%s%.1fwind_decision_min_hatB_trivialfirst{attacked}"% (bound, emi_type,windratio), decision_cost_min_hatB_trivialfirst)
        # np.save(f"{data_file}/%.2f%s%.1fwind_netload_to_print{attacked}"% (bound, emi_type,windratio), load-wind)
        np.save(f"{data_file}/%.2f%s%.1fwind_decision_theory{attacked}"% (bound, emi_type,windratio), decision_cost_hatB_trivialfirst)
        return emi_no,emi_trivial,emi_cost_min_hatB_trivialfirst,emi_cost_hatB_trivialfirst
    # record_improvement_battery_cost_min=np.array(record_improvement_battery_cost_min)
    # record_improvement_battery_cost_mintheta=np.array(record_improvement_battery_cost_mintheta)
    # record_improvement_trivial=np.array(record_improvement_trivial)
    # record_improvement_opt=np.array(record_improvement_opt)

    # record_improvement_battery_cost_min_pc=np.array(record_improvement_battery_cost_min_pc)
    # record_improvement_battery_cost_mintheta_pc=np.array(record_improvement_battery_cost_mintheta_pc)
    # record_improvement_trivial_pc=np.array(record_improvement_trivial_pc)
    # record_improvement_opt_pc=np.array(record_improvement_opt_pc)

    # record_emi_no=np.array(record_emi_no)
    # record_emi_opt=np.array(record_emi_opt)
    # record_emi_battery_cost_min=np.array(record_emi_battery_cost_min)
    # record_emi_battery_cost_mintheta=np.array(record_emi_battery_cost_mintheta)
    # record_emi_trivial=np.array(record_emi_trivial)

    # np.save(f"{data_file}/%.2f%semi_battery_cost_min"% (bound, emi_type), record_emi_battery_cost_min)
    # np.save(f"{data_file}/%.2f%semi_battery_cost_mintheta"% (bound, emi_type), record_emi_battery_cost_mintheta)
    # np.save(f"{data_file}/%.2f%semi_trivial"% (bound, emi_type), record_emi_trivial)
    # np.save(f"{data_file}/%.2f%semi_no"% (bound, emi_type), record_emi_no)
    # np.save(f"{data_file}/%.2f%semi_opt"% (bound, emi_type), record_emi_opt)

    # np.save(f"{data_file}/%.2f%simprovement_battery_cost_min"% (bound, emi_type), record_improvement_battery_cost_min)
    # np.save(f"{data_file}/%.2f%simprovement_battery_cost_mintheta"% (bound, emi_type), record_improvement_battery_cost_mintheta)
    # np.save(f"{data_file}/%.2f%simprovement_trivial"% (bound, emi_type), record_improvement_trivial)
    # np.save(f"{data_file}/%.2f%simprovement_opt"% (bound, emi_type), record_improvement_opt)

    # np.save(f"{data_file}/%.2f%simprovement_pc_battery_cost_min"% (bound, emi_type), record_improvement_battery_cost_min_pc)
    # np.save(f"{data_file}/%.2f%simprovement_pc_battery_cost_mintheta"% (bound, emi_type), record_improvement_battery_cost_mintheta_pc)
    # np.save(f"{data_file}/%.2f%simprovement_pc_trivial"% (bound, emi_type), record_improvement_trivial_pc)
    # np.save(f"{data_file}/%.2f%simprovement_pc_opt"% (bound, emi_type), record_improvement_opt_pc)
    # return 9e8,9e8,emi_trivial, 9e8, 9e8

    # return emi_opt, emi_no, emi_trivial, emi_battery_cost_min, emi_battery_cost_mintheta, emi_online_opt,emi_trivial_secondphase, emi_cost_cmh, emi_cost_cmh_trivialfirst, emi_cost_min_hatB, emi_cost_min_trivialfirst, emi_cost_min_hatB_trivialfirst
  
OSC_th_list = []
opt_list, no_list, trivial_list, mn_list, mn_theta_list =[], [], [], [], []
fake_opt_list, ts_list, cmh_list, cmh_tf_list, mn_hatB_list, mn_tf_list,mn_hatB_tf_list = [], [], [], [], [], [], []

for bound in bound_list:
    no,trivial, cost_min_hatB_trivialfirst, osc_th = run(bound/100.0)
    OSC_th_list.append(osc_th)
    # opt_list.append(opt)
    no_list.append(no)
    trivial_list.append(trivial)
    # mn_list.append(mn)
    # mn_theta_list.append(mn_theta)
    # fake_opt_list.append(fake_opt)
    # ts_list.append(trivial_secondphase)
    # cmh_list.append(cost_cmh)
    # cmh_tf_list.append(cost_cmh_trivialfirst)
    # mn_hatB_list.append(cost_min_hatB)
    # mn_tf_list.append(cost_min_trivialfirst)
    mn_hatB_tf_list.append(cost_min_hatB_trivialfirst)
    # result = pd.DataFrame({"no battery":no_list,"trivial":trivial_list,"osc barB":mn_list,"osc best theta barB":mn_theta_list,"trivial second":ts_list, "cmh":cmh_list,"cmh trivial first":cmh_tf_list, "best theta hat B":mn_hatB_list,"best theta barB trivial first":mn_tf_list,"best theta hat B trivialfirst":mn_hatB_tf_list, "opt":opt_list,"fake opt":fake_opt_list})
result  = pd.DataFrame({"no battery":no_list,"trivial":trivial_list,"best theta hat B trivialfirst":mn_hatB_tf_list,"theory": OSC_th_list})
result.to_csv(f"result_test3_wo_opt_theory_{adv_attack}_{prepath}_{mixed}_{args.net}{attacked}.csv")
# print(data_file, no_list,trivial_list,mn_list,mn_theta_list,ts_list, cmh_list,cmh_tf_list, mn_hatB_list,mn_tf_list,mn_hatB_tf_list, opt_list,fake_opt_list)


# # original_cost = 896229508
# for bound in bound_list:
#     (opt, no, trivial, mn, mn_theta,fake_opt, trivial_secondphase, cost_cmh, cost_cmh_trivialfirst, cost_min_hatB, cost_min_trivialfirst, cost_min_hatB_trivialfirst) = run(bound/100.0)
#     opt_list.append(opt)
#     no_list.append(no)
#     trivial_list.append(trivial)
#     mn_list.append(mn)
#     mn_theta_list.append(mn_theta)
#     fake_opt_list.append(fake_opt)
#     ts_list.append(trivial_secondphase)
#     cmh_list.append(cost_cmh)
#     cmh_tf_list.append(cost_cmh_trivialfirst)
#     mn_hatB_list.append(cost_min_hatB)
#     mn_tf_list.append(cost_min_trivialfirst)
#     mn_hatB_tf_list.append(cost_min_hatB_trivialfirst)
#     result = pd.DataFrame({"no battery":no_list,"trivial":trivial_list,"osc barB":mn_list,"osc best theta barB":mn_theta_list,"trivial second":ts_list, "cmh":cmh_list,"cmh trivial first":cmh_tf_list, "best theta hat B":mn_hatB_list,"best theta barB trivial first":mn_tf_list,"best theta hat B trivialfirst":mn_hatB_tf_list, "opt":opt_list,"fake opt":fake_opt_list})
#     result.to_csv(f"result_{adv_attack}_{prepath}_{mixed}_{args.net}{attacked}.csv")
# print(data_file, no_list,trivial_list,mn_list,mn_theta_list,ts_list, cmh_list,cmh_tf_list, mn_hatB_list,mn_tf_list,mn_hatB_tf_list, opt_list,fake_opt_list)
