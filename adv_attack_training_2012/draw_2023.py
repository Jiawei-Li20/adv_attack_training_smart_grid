import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
import pandas as pd
import os
from datasets.data import *
from scipy.io import loadmat
import ipdb
import seaborn as sns
import argparse
from hyperparameters import loadratio, windratio,solar_ratio
from datasets.data import *

Final = True
def to_percent(temp, position):
    return '%1.0f'%(temp) + '%'
sns.set_style("whitegrid")
# x = np.array(range(0,2.1,0.1))   
# BASE = 1920049021.9069128
base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
BASE = np.sum(np.load(base_path))
start=6024
length=2728
# length = 10
end=start+length
emi_type = 'mc'
eps = 3
m=loadmat("QuantileSolutionRampStatic161028.mat")
wind=np.array(m['wexp']).reshape(-1)
solar=np.load("changedsolar.npy")*solar_ratio#16000
renewable=windratio*(wind+solar)
renewable = renewable[start:end]
parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int, default=0)
plt.style.use('seaborn-whitegrid')
args = parser.parse_args()
M = {"adv_pred_attack_pred":r"$\hat D_t$" , "adv_pred_attack_pred_net":r"$\hat D_t^{net}$",
     "adv_cost_attack_cost":r"$\hat D_t$, with $\mathcal{C}_{{ahead}}$",
     "adv_cost_attack_cost_net":r"$\hat D_t^{net}$, with $\mathcal{C}_{{ahead}}$",
    #  "adv_cost_strg_attack_cost_strg_net":r"$\hat D_t^{net}$, with $\mathcal{C}_{{ahead}}$ and storage",
     "adv_cost_strg_strategy_attack_cost_strg_strategy_net":r"$\hat D_t^{net}$, with modified $\mathcal{C}$ and storage"}
prepath = "load_attack_bound_wo_advtrain"
Result_noadvtrain = {"adv_pred_attack_pred":f"result_wo_opt_theory_pred_{prepath}_0_False.csv",
     "adv_cost_attack_cost":f"result_wo_opt_theory_cost_{prepath}_0_False.csv",
     "adv_pred_attack_pred_net":f"result_wo_opt_theory_pred_{prepath}_0_True.csv",
     "adv_cost_attack_cost_net":f"result_wo_opt_theory_cost_{prepath}_0_True.csv",
    #  "adv_cost_strg_attack_cost_strg_net":f"result_cost_strg_wo_opt_theory_{prepath}_0_True.csv",
     "adv_cost_strg_strategy_attack_cost_strg_strategy_net":f"result_wo_opt_theory_cost_strg_strategy_{prepath}_0_True.csv"}
prepath = "test_with_plain_model_load_attack_bound"
Result_advtrain = {"adv_pred_attack_pred":f"result_wo_opt_theory_pred_{prepath}_0_Falseattacked.csv",
     "adv_cost_attack_cost":f"result_wo_opt_theory_cost_{prepath}_0_Falseattacked.csv",
     "adv_pred_attack_pred_net":f"result_wo_opt_theory_pred_{prepath}_0_Trueattacked.csv",
    #  "adv_cost_attack_cost_net":f"result_cost_{prepath}_0_Trueattacked.csv",
     "adv_cost_attack_cost_net":f"result_wo_opt_theory_cost_{prepath}_0_Trueattacked.csv",     
    #  "adv_cost_strg_attack_cost_strg_net":f"result_cost_strg_{prepath}_0_Trueattacked.csv",
     "adv_cost_strg_strategy_attack_cost_strg_strategy_net":f"result_wo_opt_theory_cost_strg_strategy_{prepath}_0_Trueattacked.csv"}
heatmap_list = ["adv_pred_attack_pred_net", "adv_cost_attack_cost_net","adv_cost_strg_strategy_attack_cost_strg_strategy_net"]
colors = ["blue","orange",'green','red','purple','brown']
path = os.path.join(os.path.abspath('.'), 'datasets/newdata.csv')  # Directory for dataset
for i,data_file in enumerate(M.keys()):
    mae = np.load(data_file+"/attacked_mae.npy")[0]
    df = pd.read_csv(Result_noadvtrain[data_file])
    # if data_file == "adv_cost_strg_attack_cost_strg_net":
    #     ipdb.set_trace()
    cost = df["no battery"][0]
    # cost = 1
    print(data_file,mae,cost)
    plt.scatter(mae, cost,color = colors[i],label = M[data_file])
plt.scatter(0, BASE, color = "black", label = "actual load")
plt.legend(fontsize = 15)
plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.xticks(fontsize =13)
plt.yticks(fontsize =13)
plt.xlabel("Prediction MAPE",fontsize = 20)
plt.ylabel("Cost($)",fontsize = 20)
if Final:
    plt.savefig("figs/4_pred.pdf",bbox_inches = "tight") 
else:
    plt.savefig("figs/4_pred.jpg",bbox_inches = "tight") 

plt.clf()
def averaged_load(l,h):
    new = []
    i=0
    while i < l.shape[0]:
        if i+h < l.shape[0]:
            new.append(np.average(l[i:i+h]))
        else:
            new.append(np.average(l[i:]))
        i = i+h
    return new


def draw_heatmap(cost_path,base_path,save_path):
    cost_trivial = np.load(cost_path)
    cost_ori_trivial = np.load(base_path)
    # import ipdb;ipdb.set_trace()
    # assert np.abs(np.sum(cost_ori_trivial))<1
    ratio_mean = np.zeros(2728)
    # zeros = 0
    mean = np.mean(np.array(cost_ori_trivial))
    for t in range(len(cost_trivial)):
        ratio_mean[t] = (cost_trivial[t]-cost_ori_trivial[t])/mean
    # ipdb.set_trace()
    data = ratio_mean[:(2728//24)*24]
    data = data.reshape(-1,24)
    pd.DataFrame(data) #以一个数据框的格式来显示
    # import ipdb;ipdb.set_trace()
    # data = np.array(data)
    # print(np.min(data),np.max(data))
    # plt.rc('axes', labelsize=20) 
    # import ipdb;ipdb.set_trace()
    num_ticks = 10
    # the index of the position of yticks
    # depth_list = np.arange(data.shape[0])
    # yticks = np.linspace(0, len(depth_list)-1, num_ticks, dtype=np.int)
    # # the content of labels of these yticks
    # yticklabels = [depth_list[idx] for idx in yticks]

    f,ax = plt.subplots(figsize=(8,14)) #定义一个子图宽高为9和6 ax存储的是图形放在哪个位置
    # ax = sns.heatmap(data, cmap = "bwr",center = 0,yticklabels=True)
    ax = sns.heatmap(data,vmin=-0.45,vmax=0.65, cmap = "bwr",center = 0,yticklabels=True) #12
    # ax = sns.heatmap(data,vmin=-0.5,vmax=1.2, cmap = "bwr",center = 0,yticklabels=True) #23

    # ax = sns.heatmap(data,vmin=-4,vmax=6, cmap = "bwr",center = 0) #40 [-5.5,6]
    # ax.set_yticks(yticks)

    # ax = sns.heatmap(data, cmap = "bwr",center = 0,yticklabels=True)
    # ax.set( xlabel = "Hours", ylabel = "Days")
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=13)

    # plt.yticks(np.arange(0,112, 4))
    # ax.tick_params(axis='both', labelsize=13, labelrotation=45)
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=35)

    # ax = sns.heatmap(data, cmap = "bwr",center = 0,yticklabels=False)
    # plt.colorbar(ticks = 0.1*np.array(list(range(-8,20,2))))
    # plt.axes().get_yaxis().set_visible(False)
    # plt.legend()
    plt.xlabel('Hours',fontsize = 35)
    plt.ylabel('Days', fontsize = 35)

    # plt.xlabel('load-wind',fontsize=18)
    # plt.ylabel('Cost Improvement ratio mean',fontsize=18)
    # plt.xlabel('Cost Improvement ratio mean',fontsize=18)
    # plt.xticks(fontsize =10)
    # plt.yticks(fontsize =10)
    plt.tight_layout()
    # plt.savefig(f"figs/cost improvement ratio mean vs load-wind_bound{bound}_base{ori}(3,6.5).png")
    # plt.savefig(f"figs/cost improvement ratio mean_bound{bound}_base{ori}(3,battery).png")
    plt.savefig(save_path,bbox_inches = "tight")

    plt.close()

h = 50

time = np.arange(start,end,h)
plt.rc('axes', labelsize=20) 
fig, ax = plt.subplots()
fig.set_size_inches(16,4)

for i,data_file in enumerate(M.keys()):
    forecast = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.00_result.csv',dtype=float,delimiter=',')
    # import ipdb;ipdb.set_trace()
    if data_file[-3:]=="net": 
        print(data_file)
        forecast = forecast[:,0].reshape(-1)+ renewable.reshape(-1)
    else:
        forecast = forecast[:,0].reshape(-1)
    forecast= averaged_load(forecast,h)
    ax.plot(time, forecast,color = colors[i], label = M[data_file])

df = load_dataset(path=path)
load = np.array(loadratio*df['actual']).reshape(-1)
ax.plot(time, averaged_load(load[start:end],h), color = "black",  label = "Actual Load")
ax.set(xlabel = "Time(hour)", ylabel = "Load Forecast(MWh)")
# import ipdb;ipdb.set_trace()
ax.tick_params(axis='both', labelsize=13)
# ax.get_legend().remove()
if Final:
    fig.savefig("figs/pred_load.pdf",bbox_inches = "tight")
else: 
    fig.savefig("figs/pred_load.jpg",bbox_inches = "tight") 


def export_legend(ax, filename):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=10,)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
if Final:
    export_legend(ax, "figs/legend.pdf")
else:
    export_legend(ax, "figs/legend.jpg")
# ax.get_legend().remove()
plt.clf()


eps = 3

time = np.arange(start,end,h)
for i,data_file in enumerate(M.keys()):
    forecast = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.0{eps}_result.csv',dtype=float,delimiter=',')
    # import ipdb;ipdb.set_trace()
    if data_file[-3:]=="net": 
        print(data_file)
        forecast = forecast[:,0].reshape(-1)+ renewable.reshape(-1)
    else:
        forecast = forecast[:,0].reshape(-1)
    forecast= averaged_load(forecast,h)
    plt.plot(time, forecast,color = colors[i], label = M[data_file])
df = load_dataset(path=path)
load = np.array(loadratio*df['actual']).reshape(-1)
plt.plot(time, averaged_load(load[start:end],h), color = "black", label = "actual load")
# plt.legend(fontsize = 6)
plt.xlabel("Time(hour)",fontsize = 20)
plt.ylabel("Attacked Load Forecast(MWh)",fontsize = 20)
plt.xticks(fontsize =10)
plt.yticks(fontsize =10)
if Final:
    plt.savefig("figs/attacked_load.pdf",bbox_inches = "tight")
else:
    plt.savefig("figs/attacked_load.jpg",bbox_inches = "tight") 

s = 8000
t = 8500
time = np.arange(s,t,1)
df = load_dataset(path=path)
load = np.array(loadratio*df['actual']).reshape(-1)
for data_file in ["adv_pred_attack_pred_net"]:
    plt.clf()
    plt.figure(figsize = (16,4))
    plt.plot(time, load[s:t], color = "black", alpha = 0.5, linestyle ='dashed', label = "actual load")
    attacked = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.0{eps}_result.csv',dtype=float,delimiter=',')
    # import ipdb;ipdb.set_trace()
    if data_file[-3:]=="net": 
        print(data_file)
        attacked = attacked[:,0].reshape(-1)+ renewable.reshape(-1)
    else:
        attacked = attacked[:,0].reshape(-1)
    # forecast= averaged_load(forecast,h)
    forecast = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.00_result.csv',dtype=float,delimiter=',')
    # import ipdb;ipdb.set_trace()
    if data_file[-3:]=="net": 
        print(data_file)
        forecast = forecast[:,0].reshape(-1)+ renewable.reshape(-1)
    else:
        forecast = forecast[:,0].reshape(-1)
    # forecast= averaged_load(forecast,h)
    plt.plot(time, forecast[s-start:t-start] ,color = 'blue', alpha = 0.5, label = "w/o attack")
    plt.plot(time, attacked[s-start:t-start] ,color = 'r', alpha = 0.5, label = "attacked")
    plt.legend(fontsize = 15)
    plt.xlabel("Time(hour)",fontsize = 20)
    plt.ylabel("Load Forecast(MWh)",fontsize = 20)
    plt.xticks(fontsize =13)
    plt.yticks(fontsize =13)
    if Final:
        plt.savefig(f"figs/load_forecast_attacked{data_file[4:]}.pdf",bbox_inches = "tight")
    else:
        plt.savefig(f"figs/load_forecast_attacked{data_file[4:]}.jpg",bbox_inches = "tight") 

plt.clf()


time = np.arange(6200,6700,1)
df = load_dataset(path=path)
load = np.array(loadratio*df['actual']).reshape(-1)
plt.plot(time, load[time], color = "black", alpha = 0.5, linestyle = 'dashed',label = "actual load")

for data_file in {"adv_pred_attack_pred","adv_cost_attack_cost"}:
    # attacked = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.0{eps}_result.csv',dtype=float,delimiter=',')
    # # import ipdb;ipdb.set_trace()
    # if data_file[-3:]=="net": 
    #     print(data_file)
    #     attacked = attacked[:,0].reshape(-1)+ renewable.reshape(-1)
    # else:
    #     attacked = attacked[:,0].reshape(-1)
    # # forecast= averaged_load(forecast,h)
    # plt.plot(time, attacked ,color = 'r', label = "Attacked")
    forecast = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.00_result.csv',dtype=float,delimiter=',')
    # import ipdb;ipdb.set_trace()
    if data_file[-3:]=="net": 
        print(data_file)
        forecast = forecast[:,0].reshape(-1)+ renewable.reshape(-1)
    else:
        forecast = forecast[:,0].reshape(-1)
    # forecast= averaged_load(forecast,h)
    plt.plot(time, forecast[time-start], label = data_file[4:8],alpha= 0.6)
    # plt.plot(time, forecast ,color = 'blue', label = "W/o attacked")
plt.legend(fontsize = 15)
plt.xlabel("Time(hour)",fontsize = 20)
plt.ylabel("Load Forecast(MWh)",fontsize = 20)
plt.xticks(fontsize =13)
plt.yticks(fontsize =13)
if Final:
    plt.savefig(f"figs/cost_to_load.pdf",bbox_inches = "tight")
else:
    plt.savefig(f"figs/cost_to_load.jpg",bbox_inches = "tight") 

plt.clf()

time = np.arange(8000,8500)
df = load_dataset(path=path)
load = np.array(loadratio*df['actual']).reshape(-1)
for data_file in ['adv_pred_attack_pred_net']:
    plt.clf()
    attacked = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.0{eps}_result.csv',dtype=float,delimiter=',')
    forecast = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.00_result.csv',dtype=float,delimiter=',')
    advtrain = np.genfromtxt(f'{data_file}/test_with_plain_model_load_attack_bound0.0{eps}_result.csv',dtype=float,delimiter=',')

    if data_file[-3:]=="net": 
        print(data_file)
        attacked = attacked[:,0].reshape(-1)+ renewable.reshape(-1)
        advtrain = advtrain[:,0].reshape(-1)+ renewable.reshape(-1)
        forecast = forecast[:,0].reshape(-1)+ renewable.reshape(-1)
    else:
        attacked = attacked[:,0].reshape(-1)
        advtrain = advtrain[:,0].reshape(-1)
        forecast = forecast[:,0].reshape(-1)
    # forecast= averaged_load(forecast,h)
    plt.plot(time, load[time], color = "black", alpha= 0.5, label = "actual load")
    plt.plot(time, forecast[time-start] ,color = 'blue', alpha= 0.5, label = "w/o attack")
    plt.plot(time, attacked[time-start] ,color = 'r', alpha= 0.5, label = "attacked")
    plt.plot(time, advtrain[time-start] ,color = 'g', alpha= 0.5, label = "adversarially trained")
    plt.legend(fontsize = 15)
    plt.xlabel("Time(hour)",fontsize = 20)
    plt.ylabel("Load Forecast(MWh)",fontsize = 20)
    plt.xticks(fontsize =13)
    plt.yticks(fontsize =13)
    if Final:
        plt.savefig(f"figs/load_forecast_advtrain_{data_file[4:]}.pdf",bbox_inches = "tight")
    else:
        plt.savefig(f"figs/load_forecast_advtrain_{data_file[4:]}.jpg",bbox_inches = "tight") 

plt.clf()

emi_type = "mc"
for data_file in heatmap_list:
    cost_attacked_trivial_strg_path = f"{data_file}/%.2f%s%.1fwind_emi_array_trivial.npy"% (eps/100.0, emi_type,windratio)
    cost_attacked_path = f"{data_file}/%.2f%s%.1fwind_emi_array_no.npy"% (eps/100.0, emi_type,windratio)
    cost_advtrain_path = f"{data_file}/%.2f%s%.1fwind_emi_array_noattacked.npy"% (eps/100.0, emi_type,windratio)

    # cost_path = "results_2022/%.2f%s%.1fwind_emi_array_no.npy"% (5/100.0, emi_type,windratio)
    base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    # BASE = np.sum(np.load(base_path))
    # base_path = "data_path/%.2f%s%.1fwind_emi_array_no.npy"% (0/100.0, emi_type,windratio)
    if Final:
        draw_heatmap(cost_attacked_path,base_path,f"figs/heatmap_attacked{eps}_{data_file[4:]}.pdf")
    else:
        draw_heatmap(cost_attacked_path,base_path,f"figs/heatmap_attacked{eps}_{data_file[4:]}.jpg")

plt.clf()
plt.rc('axes', labelsize=12) 
fig = plt.figure()
ax = fig.add_subplot()
# ax2 = ax.twinx()
for i,data_file in enumerate(M.keys()):
    scaler = np.arange(11)/100.0
    # mae = np.load(data_file+"/attacked_mae.npy")
    df = pd.read_csv(Result_noadvtrain[data_file])
    cost = np.array(df["no battery"]).reshape(-1)
    # import ipdb;ipdb.set_trace()
    # cost = 1
    print(cost)
    # ax.plot(scaler, cost, color = colors[i], label = M[data_file])
    ax.plot(scaler, (cost)/BASE, color = colors[i], marker = 'x', label = M[data_file])

# ax.legend()

ax.set_xlabel("Attack Ratio",fontsize = 20)
# ax.set_ylabel("Cost($)")
ax.set_ylabel("Cost / Baseline Cost",fontsize = 20)
fig.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

# ax2.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.xticks(fontsize =13)
plt.yticks(fontsize =13)
# ipdb.set_trace()
if Final:
    plt.savefig("figs/attacked_pred_cost.pdf",bbox_inches = "tight") 
else:   
    plt.savefig("figs/attacked_pred_cost.jpg",bbox_inches = "tight") 

# plt.clf()
# plt.rc('axes', labelsize=12) 
# fig = plt.figure()
# ax = fig.add_subplot()
# # ax2 = ax.twinx()
# for i,data_file in enumerate(M.keys()):
#     ax.plot(scaler, (cost)/BASE, color = colors[i], marker = 'x', label = M[data_file])

if Final:
    export_legend(ax,"figs/attacked_pred_cost_legend.pdf")
else:
    export_legend(ax,"figs/attacked_pred_cost_legend.jpg")
plt.clf()


# for i,data_file in enumerate(M.keys()):
#     scaler = np.arange(11)/100.0
#     scaler = np.load(data_file+"/attacked_mae.npy")

#     print(cost)
#     plt.plot(scaler, mae,color = colors[i], label = M[data_file])
# plt.legend()
# plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
# plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
# plt.xlabel("Attack Ratio")
# plt.ylabel("Prediction MAPE")
# plt.savefig("figs/attacked_pred_mape.jpg") 


#Defense part==============================================
#Cyber defense

time = np.arange(start,end,h)
plt.rc('axes', labelsize=20) 
fig, ax = plt.subplots()
for i,data_file in enumerate(M.keys()):
    forecast = np.genfromtxt(f'{data_file}/test_with_plain_model_load_attack_bound0.0{eps}_result.csv',dtype=float,delimiter=',')
    # import ipdb;ipdb.set_trace()
    if data_file[-3:]=="net": 
        print(data_file)
        forecast = forecast[:,0].reshape(-1)+ renewable.reshape(-1)
    else:
        forecast = forecast[:,0].reshape(-1)
    forecast= averaged_load(forecast,h)
    ax.plot(time, forecast,color = colors[i], label = M[data_file])
df = load_dataset(path=path)
load = np.array(loadratio*df['actual']).reshape(-1)
ax.plot(time, averaged_load(load[start:end],h), color = "black", label = "actual load")
ax.set(xlabel = "Time(hour)", ylabel = "Load Forecast(MWh)")
# import ipdb;ipdb.set_trace()
ax.tick_params(axis='both', labelsize=13)

# ax.get_legend().remove()
if Final:
    fig.savefig("figs/advtrain_eps{eps}_load.pdf",bbox_inches = "tight")
else: 
    fig.savefig("figs/advtrain_eps{eps}_load.jpg",bbox_inches = "tight") 


# def export_legend(ax, filename="figs/legend.jpg"):
#     fig2 = plt.figure()
#     ax2 = fig2.add_subplot()
#     ax2.axis('off')
#     legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=10,)
#     fig  = legend.figure
#     fig.canvas.draw()
#     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig(filename, dpi="figure", bbox_inches=bbox)
# if Final:
#     export_legend(ax, "figs/legend.pdf")
# else:
#     export_legend(ax)
# ax.get_legend().remove()
# plt.clf()
# s = 8000
# t = 8500
# time = np.arange(s,t,1)
# df = load_dataset(path=path)
# load = np.array(loadratio*df['actual']).reshape(-1)
# for i,data_file in ["adv_pred_attack_pred_net"]:
#     plt.clf()
#     plt.figure(figsize = (16,4))
#     plt.plot(time, load[s:t], color = "black", alpha = 0.5, label = "actual load")
#     attacked = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.0{eps}_result.csv',dtype=float,delimiter=',')
#     # import ipdb;ipdb.set_trace()
#     if data_file[-3:]=="net": 
#         print(data_file)
#         attacked = attacked[:,0].reshape(-1)+ renewable.reshape(-1)
#     else:
#         attacked = attacked[:,0].reshape(-1)
#     # forecast= averaged_load(forecast,h)
#     forecast = np.genfromtxt(f'{data_file}/load_attack_bound_wo_advtrain0.00_result.csv',dtype=float,delimiter=',')
#     # import ipdb;ipdb.set_trace()
#     if data_file[-3:]=="net": 
#         print(data_file)
#         forecast = forecast[:,0].reshape(-1)+ renewable.reshape(-1)
#     else:
#         forecast = forecast[:,0].reshape(-1)
#     # forecast= averaged_load(forecast,h)
#     plt.plot(time, forecast[s-start:t-start] ,color = 'blue', alpha = 0.5, label = "w/o attacked")
#     plt.plot(time, attacked[s-start:t-start] ,color = 'r', alpha = 0.5, label = "attacked")
#     plt.legend()
#     plt.xlabel("Time(hour)")
#     plt.ylabel("Load forecast")
#     if Final:
#         plt.savefig(f"figs/load_forecast_advtrain__{data_file[4:]}.pdf")
#     else:
#         plt.savefig(f"figs/load_forecast_advtrain_{data_file[4:]}.jpg") 

# plt.clf()


emi_type = "mc"
for data_file in heatmap_list:
    cost_attacked_trivial_strg_path = f"{data_file}/%.2f%s%.1fwind_emi_array_trivial.npy"% (eps/100.0, emi_type,windratio)
    cost_attacked_path = f"{data_file}/%.2f%s%.1fwind_emi_array_no.npy"% (eps/100.0, emi_type,windratio)
    cost_advtrain_path = f"{data_file}/%.2f%s%.1fwind_emi_array_noattacked.npy"% (eps/100.0, emi_type,windratio)

    base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    sys = np.load(cost_attacked_trivial_strg_path)
    att = np.load(cost_attacked_path)
    adv = np.load(cost_advtrain_path)
    base = np.load(base_path)
    # ipdb.set_trace()
    if Final:
        draw_heatmap(cost_advtrain_path,base_path,f"figs/heatmap_advtrain{eps}_{data_file[4:]}.pdf")
    else:
        draw_heatmap(cost_advtrain_path,base_path,f"figs/heatmap_advtrain{eps}_{data_file[4:]}.jpg")
plt.clf()

plt.rc('axes', labelsize=20) 
fig = plt.figure()
ax = fig.add_subplot()
# ax2 = ax.twinx()
# ax2.plot(time, temp, '-r', label = 'temp')
# ax.legend(loc=0)
# ax.grid(axis='both')
# ax.set_xlabel("Time (h)")
# ax.set_ylabel(r"Radiation ($MJ\,m^{-2}\,d^{-1}$)")
# ax2.set_ylabel(r"Temperature ($^\circ$C)")
# ax2.set_ylim(0, 35)
# ax.set_ylim(-20,100)
# ax2.legend(loc=0)
# plt.savefig('0.png')
for i,data_file in enumerate(M.keys()):
    scaler = np.arange(11)/100.0
    # mae = np.load(data_file+"/attacked_mae.npy")
    
    if os.path.exists(Result_advtrain[data_file]):
        df = pd.read_csv(Result_advtrain[data_file])
        cost = np.array(df["no battery"]).reshape(-1)
        temp = pd.read_csv(Result_noadvtrain[data_file])
        cost_0 = np.array(temp["no battery"]).reshape(-1)[0]
        cost = np.append(cost_0,cost)
        # import ipdb;ipdb.set_trace()
        # cost = 1
        print(cost)
        # ipdb.set_trace()
        # ax.plot(scaler[:cost.shape[0]], cost, color = colors[i], label = M[data_file])
        ax.plot(scaler[:cost.shape[0]], (cost)/BASE, marker = 'x',color = colors[i], label = M[data_file])
    else:
        print("="*30,Result_advtrain[data_file], "not exsits!")
# ax.legend(fontsize = 15)

ax.set_xlabel("Attack Ratio")
# ax.set_ylabel("Cost($)")
ax.set_ylabel("Cost / Baseline Cost")
fig.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
# ax.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.xticks(fontsize =13)
plt.yticks(fontsize =13)
if Final:
    plt.savefig("figs/advtrain_pred_cost.pdf",bbox_inches = "tight") 
else:   
    plt.savefig("figs/advtrain_pred_cost.jpg",bbox_inches = "tight") 


# plt.rc('axes', labelsize=12) 
# fig = plt.figure()
# ax = fig.add_subplot()
# # ax2 = ax.twinx()
# # ax2.plot(time, temp, '-r', label = 'temp')
# # ax.legend(loc=0)
# # ax.grid(axis='both')
# # ax.set_xlabel("Time (h)")
# # ax.set_ylabel(r"Radiation ($MJ\,m^{-2}\,d^{-1}$)")
# # ax2.set_ylabel(r"Temperature ($^\circ$C)")
# # ax2.set_ylim(0, 35)
# # ax.set_ylim(-20,100)
# # ax2.legend(loc=0)
# # plt.savefig('0.png')
# prepath = "test_with_plain_model_pred_data"
# plain_data_to_advtrain = {"adv_pred_attack_pred":f"result_pred_wo_opt_theory_pred_{prepath}_0_False.csv",
#      "adv_cost_attack_cost":f"result_pred_wo_opt_theory_cost_{prepath}_0_False.csv",
#      "adv_pred_attack_pred_net":f"result_pred_wo_opt_theory_pred_{prepath}_0_True.csv",
#      "adv_cost_attack_cost_net":f"result_pred_wo_opt_theory_cost_{prepath}_0_True.csv",
#     #  "adv_cost_strg_attack_cost_strg_net":f"result_cost_strg_wo_opt_theory_{prepath}_0_True.csv",
#      "adv_cost_strg_strategy_attack_cost_strg_strategy_net":f"result_pred_wo_opt_theory_cost_strg_strategy_{prepath}_0_True.csv"}

# for i,data_file in enumerate(M.keys()):
#     scaler = np.arange(11)/100.0
#     # mae = np.load(data_file+"/attacked_mae.npy")
    
#     if os.path.exists(Result_advtrain[data_file]):
#         df = pd.read_csv(Result_advtrain[data_file])
#         cost = np.array(df["no battery"]).reshape(-1)
#         temp = pd.read_csv(Result_noadvtrain[data_file])
#         cost_0 = np.array(temp["no battery"]).reshape(-1)[0]
#         cost = np.append(cost_0,cost)
#         # import ipdb;ipdb.set_trace()
#         # cost = 1
#         print(cost)
#         # ipdb.set_trace()
#         # ax.plot(scaler[:cost.shape[0]], cost, color = colors[i], label = M[data_file])
#         ax.plot(scaler[:cost.shape[0]], (cost)/BASE, marker = 'x',color = colors[i], label = M[data_file])
#         df = pd.read_csv(plain_data_to_advtrain[data_file])
#         cost = np.array(df["no battery"]).reshape(-1)
#         ax.plot(np.arange(0,11)/100.0, (cost)/BASE, marker = '+',color = colors[i], linestyle = "dashed",label = M[data_file])
#     else:
#         print("="*30,Result_advtrain[data_file], "not exsits!")

# # adv_40/result_pred_wo_opt_theory_cost_strg_strategy_test_with_plain_model_pred_data_0_True.csv

# ax.legend()

# ax.set_xlabel("Attack Ratio")
# ax.set_ylabel("Cost($)")
# # ax2.set_ylabel("Cost / Baseline Cost")
# fig.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
# # ax2.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
# plt.xticks(fontsize =10)
# plt.yticks(fontsize =10)
# if Final:
#     plt.savefig("figs/advtrain_pred_cost_vs_no_attacked_data.pdf",bbox_inches = "tight") 
# else:   
#     plt.savefig("figs/advtrain_pred_cost_vs_no_attacked_data.jpg",bbox_inches = "tight") 


plt.clf()

for i,data_file in enumerate(M.keys()):
    scale = np.arange(11)/100.0
    mae = np.load(data_file+"/attacked_mae.npy")
    # df = pd.read_csv(Result[data_file])
    # cost = df["no battery"]
    # cost = 1
    print(cost)
    if mae.shape[0]<11:
        plt.plot(scale[1:],mae,color = colors[i], marker = 'x',label = M[data_file])
    else:
        plt.plot(scale, mae,color = colors[i], marker = 'x', label = M[data_file])
# plt.legend()
plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.xlabel("Attack Ratio",fontsize = 20)
plt.ylabel("Prediction MAPE",fontsize = 20)
plt.xticks(fontsize =13)
plt.yticks(fontsize =13)
if Final:
    plt.savefig("figs/attacked_pred_mape.pdf",bbox_inches = "tight") 
else:
    plt.savefig("figs/attacked_pred_mape.jpg",bbox_inches = "tight") 

#Storage Defense ============================
for data_file in heatmap_list:
    cost_attacked_trivial_strg_path = f"{data_file}/%.2f%s%.1fwind_emi_array_trivial.npy"% (eps/100.0, emi_type,windratio)
    # cost_attacked_path = f"{data_file}/%.2f%s%.1fwind_emi_array_no.npy"% (eps/100.0, emi_type,windratio)
    # cost_advtrain_path = f"{data_file}/%.2f%s%.1fwind_emi_array_noattacked.npy"% (eps/100.0, emi_type,windratio)

    # cost_path = "results_2022/%.2f%s%.1fwind_emi_array_no.npy"% (5/100.0, emi_type,windratio)
    base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    # base_path = "data_path/%.2f%s%.1fwind_emi_array_no.npy"% (0/100.0, emi_type,windratio)
    if Final:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_trivial_storage_only{eps}_{data_file[4:]}.pdf")
    else:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_trivial_storage_only{eps}_{data_file[4:]}.jpg")
plt.clf()

for data_file in heatmap_list:
    cost_attacked_trivial_strg_path = f"{data_file}/%.2f%s%.1fwind_emi_array_min_hatB_trivialfirst.npy"% (eps/100.0, emi_type,windratio)
    # cost_attacked_path = f"{data_file}/%.2f%s%.1fwind_emi_array_no.npy"% (eps/100.0, emi_type,windratio)
    # cost_advtrain_path = f"{data_file}/%.2f%s%.1fwind_emi_array_noattacked.npy"% (eps/100.0, emi_type,windratio)

    # cost_path = "results_2022/%.2f%s%.1fwind_emi_array_no.npy"% (5/100.0, emi_type,windratio)
    base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    # base_path = "data_path/%.2f%s%.1fwind_emi_array_no.npy"% (0/100.0, emi_type,windratio)
    if Final:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_osc_storage_only{eps}_{data_file[4:]}.pdf")
    else:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_osc_storage_only{eps}_{data_file[4:]}.jpg")
plt.clf()

for data_file in heatmap_list:
    cost_attacked_trivial_strg_path = f"{data_file}/%.2f%s%.1fwind_emi_array_opt.npy"% (eps/100.0, emi_type,windratio)
    # cost_attacked_path = f"{data_file}/%.2f%s%.1fwind_emi_array_no.npy"% (eps/100.0, emi_type,windratio)
    # cost_advtrain_path = f"{data_file}/%.2f%s%.1fwind_emi_array_noattacked.npy"% (eps/100.0, emi_type,windratio)

    # cost_path = "results_2022/%.2f%s%.1fwind_emi_array_no.npy"% (5/100.0, emi_type,windratio)
    base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    # base_path = "data_path/%.2f%s%.1fwind_emi_array_no.npy"% (0/100.0, emi_type,windratio)
    if Final:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_opt_storage_only{eps}_{data_file[4:]}.pdf")
    else:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_opt_storage_only{eps}_{data_file[4:]}.jpg")
plt.clf()

for data_file in M.keys():
    scale = np.arange(11)/100.0
    df = pd.read_csv(Result_advtrain[data_file])
    advtrain  = np.array(df["no battery"]).reshape(-1)
    df = pd.read_csv(Result_noadvtrain[data_file])
    No = np.array(df["no battery"]).reshape(-1)
    advtrain = np.concatenate(([No[0]],advtrain))
    PISC = np.array(df["trivial"]).reshape(-1)
    #,no battery,trivial,osc barB,osc best theta barB,trivial second,cmh,cmh trivial first,best theta hat B,best theta barB trivial first,best theta hat B trivialfirst,opt,fake opt
    PSSC = np.array(df["best theta hat B trivialfirst"]).reshape(-1)
    if "theory" in Result_noadvtrain[data_file][7:]:
        PSSC_th = pd.read_csv(Result_noadvtrain[data_file])['theory']
    else:
        PSSC_th = pd.read_csv("result_theory_"+Result_noadvtrain[data_file][7:])['theory']
    if "fake opt" in df:
        # OPT1 = np.array(df["fake opt"]).reshape(-1)
        OPT2 = np.array(df["opt"]).reshape(-1)
    else:
        temp = pd.read_csv("result_"+Result_noadvtrain[data_file][21:])
        # ipdb;ipdb.set_trace()
        # OPT1 = np.array(temp["fake opt"]).reshape(-1)
        OPT2 = np.array(temp["opt"]).reshape(-1)
    # OPT1 = np.array(df["fake opt"]).reshape(-1)
    # OPT2 = np.array(df["opt"]).reshape(-1)
    No = (No)/BASE
    advtrain = advtrain/BASE
    PISC = (PISC)/BASE
    PSSC = (PSSC)/BASE
    PSSC_th = (PSSC_th)/BASE
    # OPT1 = (OPT1)/BASE
    OPT2 = (OPT2)/BASE
    if No.shape[0]<11:
        scale = scale[1:]
    plt.plot(scale, No,marker = '.',label="w/o storage",markersize = 8)
    plt.plot(scale, PISC, marker = 'o',label = "PISC",markersize = 8)
    plt.plot(scale, PSSC, marker = '_',label = "PSSC",markersize = 8)
    plt.plot(scale, PSSC_th, marker = '3',label = "PSSC with theoretical threshold",markersize = 8)
    plt.plot(scale[:OPT2.shape[0]], OPT2, marker = '1',label = "OPT",markersize = 8)
    # plt.plot(scale[:OPT2.shape[0]], OPT2, marker = '2', label = "OPT2",markersize = 8)
    plt.plot(scale, advtrain, marker = '2', label = "advtrain",markersize = 8)
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.xlabel("Attack Ratio",fontsize = 30)
    plt.ylabel("Cost / Baseline Cost",fontsize = 30)
    # plt.legend()
    plt.xticks(fontsize =20)
    plt.yticks(fontsize =20)
    if Final:
        plt.savefig(f"figs/Storage_defense_{data_file[4:]}.pdf",bbox_inches = "tight")
    else:
        plt.savefig(f"figs/Storage_defense_{data_file[4:]}.jpg",bbox_inches = "tight")
    # cost_path = "results_2022/%.2f%s%.1fwind_emi_array_no.npy"% (5/100.0, emi_type,windratio)
    # base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    # base_path = "data_path/%.2f%s%.1fwind_emi_array_no.npy"% (0/100.0, emi_type,windratio)
    plt.clf()
plt.rc('axes', labelsize=12) 
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(scale, No,marker = '.',label="w/o storage",markersize = 8)
ax.plot(scale, PISC, marker = 'o',label = "PISC",markersize = 8)
ax.plot(scale, PSSC, marker = '_',label = "PSSC",markersize = 8)
ax.plot(scale, PSSC_th, marker = '3',label = "PSSC with theoretical threshold",markersize = 8)
ax.plot(scale[:OPT2.shape[0]], OPT2, marker = '1',label = "OPT",markersize = 8)
# plt.plot(scale[:OPT2.shape[0]], OPT2, marker = '2', label = "OPT2",markersize = 8)
ax.plot(scale, advtrain, marker = '2', label = "advtrain",markersize = 8)

if Final:
    export_legend(ax, "figs/Storage_defense_legend.pdf")
else:
    export_legend(ax, "figs/Storage_defense_legend.jpg")
plt.clf()



# System defense=====================================
for data_file in heatmap_list:
    cost_attacked_trivial_strg_path = f"{data_file}/%.2f%s%.1fwind_emi_array_trivialattacked.npy"% (eps/100.0, emi_type,windratio)
    # cost_attacked_path = f"{data_file}/%.2f%s%.1fwind_emi_array_no.npy"% (eps/100.0, emi_type,windratio)
    # cost_advtrain_path = f"{data_file}/%.2f%s%.1fwind_emi_array_noattacked.npy"% (eps/100.0, emi_type,windratio)

    # cost_path = "results_2022/%.2f%s%.1fwind_emi_array_no.npy"% (5/100.0, emi_type,windratio)
    base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    # base_path = "data_path/%.2f%s%.1fwind_emi_array_no.npy"% (0/100.0, emi_type,windratio)
    if Final:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_advtrain_trivial_storage{eps}_{data_file[4:]}.pdf")
    else:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_advtrain_trivial_storage{eps}_{data_file[4:]}.jpg")
plt.clf()

for data_file in heatmap_list:
    cost_attacked_trivial_strg_path = f"{data_file}/%.2f%s%.1fwind_emi_array_min_hatB_trivialfirstattacked.npy"% (eps/100.0, emi_type,windratio)
    # cost_attacked_path = f"{data_file}/%.2f%s%.1fwind_emi_array_no.npy"% (eps/100.0, emi_type,windratio)
    # cost_advtrain_path = f"{data_file}/%.2f%s%.1fwind_emi_array_noattacked.npy"% (eps/100.0, emi_type,windratio)

    # cost_path = "results_2022/%.2f%s%.1fwind_emi_array_no.npy"% (5/100.0, emi_type,windratio)
    base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    # base_path = "data_path/%.2f%s%.1fwind_emi_array_no.npy"% (0/100.0, emi_type,windratio)
    if Final:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_advtrain_osc_storage{eps}_{data_file[4:]}.pdf")
    else:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_advtrain_osc_storage{eps}_{data_file[4:]}.jpg")
plt.clf()

for data_file in heatmap_list:
    cost_attacked_trivial_strg_path = f"{data_file}/%.2f%s%.1fwind_emi_array_optattacked.npy"% (eps/100.0, emi_type,windratio)
    # cost_attacked_path = f"{data_file}/%.2f%s%.1fwind_emi_array_no.npy"% (eps/100.0, emi_type,windratio)
    # cost_advtrain_path = f"{data_file}/%.2f%s%.1fwind_emi_array_noattacked.npy"% (eps/100.0, emi_type,windratio)

    # cost_path = "results_2022/%.2f%s%.1fwind_emi_array_no.npy"% (5/100.0, emi_type,windratio)
    base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    # base_path = "data_path/%.2f%s%.1fwind_emi_array_no.npy"% (0/100.0, emi_type,windratio)
    if Final:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_advtrain_opt_storage{eps}_{data_file[4:]}.pdf")
    else:
        draw_heatmap(cost_attacked_trivial_strg_path,base_path,f"figs/heatmap_advtrain_opt_storage{eps}_{data_file[4:]}.jpg")
plt.clf()

for data_file in M.keys():
    scale = np.arange(1,11)/100.0
    df = pd.read_csv(Result_advtrain[data_file])
    No = np.array(df["no battery"]).reshape(-1)
    PISC = np.array(df["trivial"]).reshape(-1)
    #,no battery,trivial,osc barB,osc best theta barB,trivial second,cmh,cmh trivial first,best theta hat B,best theta barB trivial first,best theta hat B trivialfirst,opt,fake opt
    PSSC = np.array(df["best theta hat B trivialfirst"]).reshape(-1)
    if "theory" in Result_advtrain[data_file][7:]:
        PSSC_th = pd.read_csv(Result_advtrain[data_file])['theory']
    else:
        PSSC_th = pd.read_csv("result_theory_"+Result_advtrain[data_file][7:])['theory']
    if "fake opt" in df:
        # OPT1 = np.array(df["fake opt"]).reshape(-1)
        OPT2 = np.array(df["opt"]).reshape(-1)
    else:
        temp = pd.read_csv("result_"+Result_advtrain[data_file][21:])
        # OPT1 = np.array(temp["fake opt"]).reshape(-1)
        OPT2 = np.array(temp["opt"]).reshape(-1)
    No = (No)/BASE
    PISC = (PISC)/BASE
    PSSC = (PSSC)/BASE
    PSSC_th = (PSSC_th)/BASE
    # OPT1 = (OPT1)/BASE
    OPT2 = (OPT2)/BASE
    plt.plot(scale, No,marker = '.', label="w/o storage",markersize = 8)
    plt.plot(scale, PISC, marker = 'o', label = "PISC",markersize = 8)
    plt.plot(scale, PSSC, marker = '_', label = "PSSC",markersize = 8)
    plt.plot(scale, PSSC_th, marker = '3', label = "PSSC with theoretical threshold")
    # plt.plot(scale[:OPT1.shape[0]], OPT1, marker = '1', label = "OPT1",markersize = 8)
    plt.plot(scale[:OPT2.shape[0]], OPT2, marker = '2', label = "OPT",markersize = 8)
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.xlabel("Attack Ratio",fontsize = 30)
    plt.ylabel("Cost / Baseline Cost",fontsize = 30)
    # plt.legend()
    plt.xticks(fontsize =20)
    plt.yticks(fontsize =20)

    if Final:
        plt.savefig(f"figs/System_defense_{data_file[4:]}.pdf",bbox_inches = "tight")
    else:
        plt.savefig(f"figs/System_defense_{data_file[4:]}.jpg",bbox_inches = "tight")
    # cost_path = "results_2022/%.2f%s%.1fwind_emi_array_no.npy"% (5/100.0, emi_type,windratio)
    # base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    # base_path = "data_path/%.2f%s%.1fwind_emi_array_no.npy"% (0/100.0, emi_type,windratio)
    plt.clf()
plt.rc('axes', labelsize=12) 
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(scale, No,marker = '.', label="w/o storage",markersize = 8)
ax.plot(scale, PISC, marker = 'o', label = "PISC",markersize = 8)
ax.plot(scale, PSSC, marker = '_', label = "PSSC",markersize = 8)
ax.plot(scale, PSSC_th, marker = '3', label = "PSSC with theoretical threshold")
ax.plot(scale[:OPT2.shape[0]], OPT2, marker = '2', label = "OPT",markersize = 8)
    
if Final:
    export_legend(ax, "figs/System_defense_legend.pdf")
else:
    export_legend(ax, "figs/System_defense_legend.jpg")
plt.clf()
raise ValueError
