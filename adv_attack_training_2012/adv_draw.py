
import pandas
import scipy
from scipy.io import loadmat
import math

import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
import numpy as np
precise_prediction=True
non_precise_prediction=True
prediction_1=True
prediction_5=True
prediction_10=True

# b = np.genfromtxt('solar.csv',dtype=float,delimiter=',')
# b[0]=0.09
# np.save('solar',b)
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

####emission function
emission_function=False
if emission_function==True:
    capacity=np.load("capacity.npy")
    ner=np.load("ner.npy")
    cer=np.load("cer.npy")
    cher=np.load("cher.npy")
    temptot=0
    ner_curve=np.array([0])
    n=151
    for i in range(0,n):
        for j in range(0,capacity[i]):
            temptot=temptot+ner[i]
            ner_curve=np.append(ner_curve,temptot)
            #print(plant_emi_curve)
    print("no2 tot",temptot)
    temptot=0
    ctot=0
    chtot=0
    ser=np.load("ser.npy")
    ser_curve=np.array([0])
    cer_curve=np.array([0])
    cher_curve=np.array([0])
    for i in range(0,151):
        for j in range(0,capacity[i]):
            temptot=temptot+ser[i]
            ser_curve=np.append(ser_curve,temptot)
            ctot=ctot+cer[i]
            cer_curve=np.append(cer_curve,ctot)
            chtot=chtot+cher[i]
            cher_curve=np.append(cher_curve,chtot)
    print("so2 tot",temptot)
    print("co2 tot",ctot)
    print("ch tot",chtot)

    print(len(cer_curve))
    plt.plot(ner_curve,color='red')
    plt.plot(ser_curve*0.5)
    plt.plot(cer_curve*0.0005,color='black')
    plt.plot(cher_curve*0.05,color='green')
    plt.legend(["nitrogen oxides","sulfur dioxide",'carbon dioxide','methane'])
    plt.xlabel('power(MWh)')
    plt.ylabel('emission(tons)')
    plt.title('Emission Function')
    plt.show()

####prcise prediction 
# Example Python program to plot a complex bar chart 

import pandas as pd
if precise_prediction==True:
# A python dictionary

    data = {"No Battery":[ 64029148.96910466,51760312.27310355 , 42304290.238802895, 32464586.783002157],

            "ONCSC":[55617973.89570315, 44933037.04320195,37603297.86110143 ,30417234.097702067  ],

            "ONCSC$^*$":[55592024.68600311, 44535130.28950177, 36084034.54450139, 27258520.680100683]

            };

    a='$\\rho_1$'
    index     = [a, '$\\rho_2$', '$\\rho_3$', '$\\rho_4$'];

     

    # Dictionary loaded into a DataFrame       

    dataFrame = pd.DataFrame(data=data, index=index);

     

    # Draw a vertical bar chart

    dataFrame.plot.bar(rot=15);
    #plt.title("Nitrogen Oxides Emission Under Different Algorithms")
    plt.ylabel('nitrogen oxides emission(tons)')
    plt.ylim([20000000,70000000])
    plt.show(block=True)
    #plt.savefig('results/no_emission_precise.png', dpi=150)


if non_precise_prediction==True:
# A python dictionary

    data = {"No Battery":[64692038.349404834, 52643569.99550361,43318851.76990302 , 33543257.502702225],

            "ONCSC":[56218409.02630321, 45282309.39080199,37787520.41080131 , 30576117.616602033],

            "ONCSC$^*$":[56222393.15740306, 45089348.362701826, 36648103.94050114, 27767960.988800995]

            };

    a='$\\rho_1$'
    index     = [a, '$\\rho_2$', '$\\rho_3$', '$\\rho_4$'];

     

    # Dictionary loaded into a DataFrame       

    dataFrame = pd.DataFrame(data=data, index=index);

     

    # Draw a vertical bar chart

    dataFrame.plot.bar(rot=15);
    #plt.title("Nitrogen Oxides Emission Under Different Algorithms")
    plt.ylabel('nitrogen oxides emission(tons)')
    plt.ylim([20000000,70000000])
    plt.show(block=True)
    #plt.savefig('results/no_emission_precise.png', dpi=150)

if prediction_1==True:
# A python dictionary

    data = {"No Battery":[ 64657338.908404686,52861333.971403755 ,43827668.28410302 , 34274114.44870217],

            "ONCSC":[56538472.853903405 , 45665958.441702075,37978147.50570169 , 30715961.43130204],

            "ONCSC$^*$":[ 56513074.601303324 , 45409480.38250204,36985984.69100127 ,28156018.814400904 ]

            };

    a='$\\rho_1$'
    index     = [a, '$\\rho_2$', '$\\rho_3$', '$\\rho_4$'];

     

    # Dictionary loaded into a DataFrame       

    dataFrame = pd.DataFrame(data=data, index=index);

     

    # Draw a vertical bar chart

    dataFrame.plot.bar(rot=15);
    #plt.title("Nitrogen Oxides Emission Under Different Algorithms")
    plt.ylabel('nitrogen oxides emission(tons)')
    plt.ylim([20000000,70000000])
    plt.show(block=True)
    #plt.savefig('results/no_emission_precise.png', dpi=150)
if prediction_5==True:
# A python dictionary

    data = {"No Battery":[66079463.4820045, 55238864.53520339, 47212783.933502845,38306437.8153018],

            "ONCSC":[59404895.40680333,48409519.069402315 , 40114267.84610238,31686146.556102287 ],

            "ONCSC$^*$":[59371048.21570321,47953976.39070223 , 39253094.70890143, 30460895.83520111]

            };

    a='$\\rho_1$'
    index     = [a, '$\\rho_2$', '$\\rho_3$', '$\\rho_4$'];

     

    # Dictionary loaded into a DataFrame       

    dataFrame = pd.DataFrame(data=data, index=index);

     

    # Draw a vertical bar chart

    dataFrame.plot.bar(rot=15);
    #plt.title("Nitrogen Oxides Emission Under Different Algorithms")
    plt.ylabel('nitrogen oxides emission(tons)')
    plt.ylim([20000000,70000000])
    plt.show(block=True)
    #plt.savefig('results/no_emission_precise.png', dpi=150)

if prediction_10==True:
# A python dictionary

    data = {"No Battery":[68512206.24530406, 59363016.4364030,52570882.491402395,  44793424.88050145],

            "ONCSC":[62151492.74920331,51399233.64660287,42911036.22900298,33208872.256602164],

            "ONCSC$^*$":[ 62144374.913403414,51313598.170902826,42546255.164901756,33208417.22050225]

            };

    a='$\\rho_1$'
    index     = [a, '$\\rho_2$', '$\\rho_3$', '$\\rho_4$'];

     

    # Dictionary loaded into a DataFrame       

    dataFrame = pd.DataFrame(data=data, index=index);

     

    # Draw a vertical bar chart

    dataFrame.plot.bar(rot=15);
    #plt.title("Nitrogen Oxides Emission Under Different Algorithms")
    plt.ylabel('nitrogen oxides emission(tons)')
    plt.ylim([20000000,70000000])
    plt.show(block=True)
    #plt.savefig('results/no_emission_precise.png', dpi=150)

