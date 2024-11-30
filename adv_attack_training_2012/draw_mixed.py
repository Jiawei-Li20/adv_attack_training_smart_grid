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

Final =False
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
mixed = 0.2
parser.add_argument("--id", type=int, default=0)
plt.style.use('seaborn-whitegrid')
args = parser.parse_args()
M = {"adv_pred_attack_pred":r"$\hat D_t$" , "adv_pred_attack_pred_net":r"$\hat D_t^{net}$",
     "adv_cost_attack_cost":r"$\hat D_t$, with $\mathcal{C}_{{ahead}}$",
     "adv_cost_attack_cost_net":r"$\hat D_t^{net}$, with $\mathcal{C}_{{ahead}}$",
    #  "adv_cost_strg_attack_cost_strg_net":r"$\hat D_t^{net}$, with $\mathcal{C}_{{ahead}}$ and storage",
     "adv_cost_strg_strategy_attack_cost_strg_strategy_net":r"$\hat D_t^{net}$, with modified $\mathcal{C}$ and storage"}
prepath = "load_attack_bound_wo_advtrain"
Result_noadvtrain = {"adv_pred_attack_pred":f"result_mixed{mixed}_wo_opt_theory_pred_{prepath}_0_False.csv",
     "adv_cost_attack_cost":f"result_mixed{mixed}_wo_opt_theory_cost_{prepath}_0_False.csv",
     "adv_pred_attack_pred_net":f"result_mixed{mixed}_wo_opt_theory_pred_{prepath}_0_True.csv",
     "adv_cost_attack_cost_net":f"result_mixed{mixed}_wo_opt_theory_cost_{prepath}_0_True.csv",
    #  "adv_cost_strg_attack_cost_strg_net":f"result_cost_strg_wo_opt_theory_{prepath}_0_True.csv",
     "adv_cost_strg_strategy_attack_cost_strg_strategy_net":f"result_mixed{mixed}_wo_opt_theory_cost_strg_strategy_{prepath}_0_True.csv"}
prepath = "test_with_plain_model_load_attack_bound"
Result_advtrain = {"adv_pred_attack_pred":f"result_mixed{mixed}_wo_opt_theory_pred_{prepath}_0_Falseattacked.csv",
     "adv_cost_attack_cost":f"result_mixed{mixed}_wo_opt_theory_cost_{prepath}_0_Falseattacked.csv",
     "adv_pred_attack_pred_net":f"result_mixed{mixed}_wo_opt_theory_pred_{prepath}_0_Trueattacked.csv",
    #  "adv_cost_attack_cost_net":f"result_cost_{prepath}_0_Trueattacked.csv",
     "adv_cost_attack_cost_net":f"result_mixed{mixed}_wo_opt_theory_cost_{prepath}_0_Trueattacked.csv",     
    #  "adv_cost_strg_attack_cost_strg_net":f"result_cost_strg_{prepath}_0_Trueattacked.csv",
     "adv_cost_strg_strategy_attack_cost_strg_strategy_net":f"result_mixed{mixed}_wo_opt_theory_cost_strg_strategy_{prepath}_0_Trueattacked.csv"}
heatmap_list = ["adv_pred_attack_pred_net", "adv_cost_attack_cost_net","adv_cost_strg_strategy_attack_cost_strg_strategy_net"]
colors = ["blue","orange",'green','red','purple','brown']
path = os.path.join(os.path.abspath('.'), 'datasets/newdata.csv')  # Directory for dataset
plt.clf()

for data_file in M.keys():
    scale = np.arange(11)/100.0
    df = pd.read_csv(Result_advtrain[data_file])
    df_0 = pd.read_csv(Result_noadvtrain[data_file])
    No = np.concatenate(np.array(df["no battery"]).reshape(-1),np.array(df_0["no battery"]).reshape(-1)[0])
    PISC = np.concatenate(np.array(df["trivial"]).reshape(-1),np.array(df_0["trivial"]).reshape(-1)[0])
    #,no battery,trivial,osc barB,osc best theta barB,trivial second,cmh,cmh trivial first,best theta hat B,best theta barB trivial first,best theta hat B trivialfirst,opt,fake opt
    PSSC = np.concatenate(np.array(df["best theta hat B trivialfirst"]).reshape(-1),np.array(df_0["best theta hat B trivialfirst"]).reshape(-1)[0])
    PSSC_th = np.concatenate(pd.read_csv(Result_advtrain[data_file])['theory'],np.array(df_0["theory"]).reshape(-1)[0])
 
    No = (No)/BASE
    PISC = (PISC)/BASE
    PSSC = (PSSC)/BASE
    PSSC_th = (PSSC_th)/BASE

    plt.plot(scale, No,marker = '.', label="w/o storage",markersize = 8)
    plt.plot(scale, PISC, marker = 'o', label = "PISC",markersize = 8)
    plt.plot(scale, PSSC, marker = '_', label = "PSSC",markersize = 8)
    if PSSC_th.shape[0]>10:
        ipdb.set_trace()
    plt.plot(scale, PSSC_th, marker = '3', label = "PSSC with theoretical threshold")
    # plt.plot(scale[:OPT1.shape[0]], OPT1, marker = '1', label = "OPT1",markersize = 8)
    # plt.plot(scale[:OPT2.shape[0]], OPT2, marker = '2', label = "OPT2",markersize = 8)
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.xlabel("Attack Ratio")
    plt.ylabel("Cost / Baseline Cost")
    plt.legend()
    plt.xticks(fontsize =10)
    plt.yticks(fontsize =10)

    if Final:
        plt.savefig(f"figs/System_defense_{data_file[4:]}.pdf",bbox_inches = "tight")
    else:
        plt.savefig(f"figs/System_defense_{data_file[4:]}.jpg",bbox_inches = "tight")
    # cost_path = "results_2022/%.2f%s%.1fwind_emi_array_no.npy"% (5/100.0, emi_type,windratio)
    # base_path = f"adv_pred_attack_pred/%.2f%s%.1fwind_emi_array_no.npy"% (-0.01, "mc",windratio)
    # base_path = "data_path/%.2f%s%.1fwind_emi_array_no.npy"% (0/100.0, emi_type,windratio)
    
    plt.clf()
raise ValueError
