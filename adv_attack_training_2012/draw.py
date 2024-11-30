import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'
def read(bound):
    # record_emi_battery_cost_min = np.load("results/%.2f%semi_battery_cost_min.npy"% (bound, emi_type))
    record_emi_trivial = np.load("results/%.2f%semi_trivial.npy"% (bound, emi_type))
    record_emi_no = np.load("results/%.2f%semi_no.npy"% (bound, emi_type))
    record_improvement_trivial_pc= np.load("results/%.2f%simprovement_pc_trivial.npy"% (bound, emi_type))
    record_emi_battery_cost_min = None
    return record_emi_battery_cost_min,record_emi_trivial,record_emi_no,record_improvement_trivial_pc

    # np.load("%.2f%simprovement_battery_cost_min.npy"% (bound, emi_type), record_improvement_battery_cost_min)
    # np.load("%.2f%simprovement_trivial.npy"% (bound, emi_type), record_improvement_trivial)

    # np.load("%.2f%simprovement_pc_battery_cost_min.npy"% (bound, emi_type), record_improvement_battery_cost_min_pc)

cost = np.array([])
print(len(cost))
no = np.array([])
tri = np.array([])
emi_type = "mc"
x=[]

for bound in range(-1,11,1):
    # print(cost.shape)
    # print(tri.shape)
    # print(no.shape)
    emi_cost, emi_tri,emi_no,imp_trivial = read(bound/100.0)
    x.append(bound/100.0)
    if len(no)==0:
        # cost = emi_cost.reshape(1,4)
        tri = emi_tri.reshape(1,4)
        no = emi_no.reshape(1,4)
        imp_tri = imp_trivial.reshape(1,4)
    else:
        # cost = np.concatenate((cost,emi_cost.reshape(1,4)),axis = 0)
        tri = np.concatenate((tri,emi_tri.reshape(1,4)),axis = 0)
        no = np.concatenate((no,emi_no.reshape(1,4)),axis = 0)
        imp_tri = np.concatenate((imp_tri,imp_trivial.reshape(1,4)),axis = 0)

print(imp_tri)
# plt.plot(cost, marker='o', mec='r', mfc='w')
# print(cost.shape)
print(tri.shape)
print(no.shape)
print(len(x))
print(x)
# print(emi_cost.shape)

import seaborn as sns
import argparse

sns.set_style("whitegrid")
# x = np.array(range(0,2.1,0.1))    
parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int, default=0)

args = parser.parse_args()

id = args.id
plt.figure(figsize=(8,4),dpi=150)
# plt.plot(x,cost[:,id],label="battery_ONCSC*")  
plt.plot(x,tri[:,id],label="battery_trivial")  
plt.plot(x,no[:,id],label="no battery")  

plt.legend()
    # plt.yscale('log')
#     plt.xlim(0,2000)
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

plt.xlabel(f'ratio bound of change(rho_{id+1})',fontsize=18)
plt.ylabel('Cost',fontsize=18)
#     algo=ori_algo
# plt.title('Cost vs Ratio(rho_2)')
plt.tight_layout()
# plt.ylim(0,1e6)


plt.savefig(f"figs/cost(rho_{id+1}).png")
plt.close()
