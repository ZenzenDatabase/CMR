#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:22:18 2019

@author: dhzeng
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.spatial.distance import cdist

##########################################################################################
prec_ugach_av = []
rec_ugach_av = []

kk=0
with open("/home/dhzeng/TNN_CCCA/UGACH/Result/prc1", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.split()
            prec_ugach_av.append(float(pre))
            rec_ugach_av.append(float(rec))
        kk+=1
        
prec_ugach_va = []
rec_ugach_va = []

kk=0
with open("/home/dhzeng/TNN_CCCA/UGACH/Result/prc2", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.split()
            prec_ugach_va.append(float(pre))
            rec_ugach_va.append(float(rec))
        kk+=1

prec_acmr_av = []
rec_acmr_av = []

kk=0
with open("/home/dhzeng/TNN_CCCA/ACMR/result/prc1_av_cl.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.strip().split()
            prec_acmr_av.append(float(pre))
            rec_acmr_av.append(float(rec))
        kk+=1
        
prec_acmr_va = []
rec_acmr_va = []

kk=0
with open("/home/dhzeng/TNN_CCCA/ACMR/result/prc1_va_cl.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.split()
            prec_acmr_va.append(float(pre))
            rec_acmr_va.append(float(rec))
        kk+=1

#******************************dcca************************
prec_dcca_av = []
rec_dcca_av = []

kk=0
with open("/home/dhzeng/TNN_CCCA/DCCA/result/prc_av_data1.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.strip().split()
            prec_dcca_av.append(float(pre))
            rec_dcca_av.append(float(rec))
        kk+=1
        
prec_dcca_va = []
rec_dcca_va = []

kk=0
with open("/home/dhzeng/TNN_CCCA/DCCA/result/prc_va_data1.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.split()
            prec_dcca_va.append(float(pre))
            rec_dcca_va.append(float(rec))
        kk+=1
#******************************ucal************************
prec_ucal_av = []
rec_ucal_av = []

kk=0
with open("/home/dhzeng/TNN_CCCA/ACMR/result/prc_av_ucal_data1.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.strip().split()
            prec_ucal_av.append(float(pre))
            rec_ucal_av.append(float(rec))
        kk+=1
        
prec_ucal_va = []
rec_ucal_va = []

kk=0
with open("/home/dhzeng/TNN_CCCA/ACMR/result/prc_va_ucal_data1.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.split()
            prec_ucal_va.append(float(pre))
            rec_ucal_va.append(float(rec))
        kk+=1
#****************************agah************************
prec_agah_av = []
rec_agah_av = []

kk=0
with open("/home/dhzeng/TNN_CCCA/AGAH/result/prc_av_data1.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.strip().split()
            prec_agah_av.append(float(pre))
            rec_agah_av.append(float(rec))
        kk+=1
        
prec_agah_va = []
rec_agah_va = []

kk=0
with open("/home/dhzeng/TNN_CCCA/AGAH/result/prc_va_data1.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.split()
            prec_agah_va.append(float(pre))
            rec_agah_va.append(float(rec))
        kk+=1
####################################################################


####################################################################
output_num = [0, 50, 100, 200, 300, 400, 500, 600, 800, 900, 1000, 1200, 1500, 1800, 2000]

## cca
prec_cca_av = [0.429, 0.4084, 0.398, 0.38333, 0.397, 0.37, 0.3666667, 0.3571429, 0.3475, 0.33,
               0.3257, 0.319333, 0.31066667, 0.28133, 0.1533333, 0.11675, 0.1196, 0.1049]
rec_cca_av  = [0.00807537, 0.01480485, 0.03364738, 0.0551817,  0.06594886, 0.07402423, 0.09555855,
               0.11170929, 0.12651413, 0.14670256, 0.17362046, 0.21265141, 0.25033647, 0.3512786,
               0.52489906, 0.69986541, 0.90040377, 0.97846568]

## dcca
#prec_dcca_av = [0.09,0.1, 0.12, 0.11333333, 0.12, 0.124, 0.12333333, 0.12285714, 0.12625, 0.13333333, 
#                0.134, 0.13, 0.132, 0.129, 0.125, 0.10675, 0.099956, 0.1049]
#rec_dcca_av  = [0.00544959, 0.01498638, 0.03814714, 0.05858311, 0.07765668, 0.09128065, 0.10626703,
#                0.11307902, 0.126703, 0.14305177, 0.15122616, 0.18119891, 0.24250681, 0.33514986, 
#                0.51907357, 0.70435967, 0.88555858, 0.97411444]

## c-cca
prec_ccca_av = [0.87, 0.89, 0.855, 0.83333333, 0.815, 0.72, 0.6333333, 0.5428571, 0.47375, 0.4222222, 
                0.37, 0.31333333, 0.25933333, 0.201, 0.1233333, 0.10975, 0.1019956, 0.1049]
rec_ccca_av  = [0.0221, 0.0811, 0.12232, 0.1743, 0.2023, 0.236,0.272,
                   0.3133, 0.343,  0.365, 0.4132, 0.443, 0.4734, 0.5143,
                   0.5932, 0.7234, 0.885, 0.9823]

## s-dcca
prec_sdcca_av = [0.915, 0.925, 0.925, 0.89333333, 0.835, 0.762, 0.65333333, 0.57428571, 0.50375, 0.45222222, 
                0.409, 0.34333333, 0.27933333, 0.211, 0.14233333, 0.1109675, 0.109956, 0.1049]
rec_sdcca_av  = [0.02234, 0.081243, 0.10132, 0.18433, 0.24534, 0.2923,0.30532,
                   0.3432, 0.3595,  0.38634, 0.42454, 0.44345, 0.4734, 0.5296,
                   0.5943, 0.7234, 0.89845, 0.984345]

## tripletNNccca
prec_tptccca_av = [0.965, 0.965, 0.95, 0.94666667,  0.91, 0.83, 0.733333, 0.62714286, 0.575,  0.49, 
                   0.4342, 0.383333, 0.36733333, 0.315, 
                   0.245, 0.12307, 0.109856,0.1049]
rec_tptccca_av  = [0.02803738, 0.08271028, 0.104205607, 0.1906308411, 0.2506542056, 0.29579439,0.30915888,
                   0.34485981, 0.3588785,  0.38925234, 0.42196262, 0.44299065, 0.450140187, 0.47383178,
                   0.53037383, 0.72429907, 0.89953271, 0.98598131]


########################################################

####################################################
####################################################
### cca

output_num = [0, 50, 100, 200, 300, 400, 500, 600, 800, 900, 1000, 1200, 1500, 1800, 2000]

prec_cca_va = [0.4091, 0.40321, 0.38965, 0.38231, 0.3732, 0.3701, 0.3536, 0.3479, 0.33975, 0.3213, 0.272, 0.24552, 0.2323,
               0.2355 ,  0.132,  0.107, 0.09856, 0.1049]
rec_cca_va  = [0.00807537, 0.01480485, 0.03364738, 0.0551817,  0.06594886, 0.07402423,
               0.09555855, 0.11170929, 0.12651413, 0.14670256, 0.17362046, 0.21265141,
               0.25033647, 0.3512786,  0.52489906, 0.69986541, 0.90040377, 0.97846568]

#### dcca
#prec_dcca_va = [0.14, 0.11, 0.105,  0.10666667, 0.115,  0.114,  0.12166667, 0.13,
#                0.125, 0.12888889, 0.126, 0.13166667, 0.13066667, 0.1395, 
#                0.13233333, 0.134, 0.1238, 0.1049]
#rec_dcca_va  = [0.00544959, 0.01498638, 0.03814714, 0.05858311, 0.07765668, 0.09128065,
#                0.10626703, 0.11307902, 0.126703,   0.14305177, 0.15122616, 0.18119891,
#                0.24250681, 0.33514986, 0.51907357, 0.70435967, 0.88555858, 0.97411444]
### c-cca
prec_ccca_va = [0.9,0.86, 0.82, 0.78333333, 0.73, 0.61, 0.566667, 0.514571429, 0.4795,
                 0.434777778, 0.409, 0.34083333, 0.27466667, 0.2085, 0.14033333, 0.107,0.09856,0.1049]
rec_ccca_va  = [0.0281, 0.0834, 0.1423, 0.1763, 0.2134, 0.2533,0.2732,
                   0.3063, 0.3365,  0.3542, 0.3753, 0.4064, 0.4534, 0.4832,
                   0.5811, 0.7213, 0.8897, 0.9821]
### s-dcca
prec_sdcca_va = [0.92, 0.91,0.87,0.82333333, 0.765, 0.694, 0.61166667, 0.54571429, 0.495,
                 0.44777778, 0.419, 0.35083333, 0.28466667, 0.2095, 
                 0.16033333, 0.109,0.109856,0.1049]
rec_sdcca_va  = [0.0281, 0.0834, 0.1123, 0.1876, 0.2324, 0.2745,0.29093,
                   0.3263, 0.3465,  0.378, 0.3932, 0.4164, 0.4634, 0.4932,
                   0.5811, 0.7213, 0.8897, 0.9821]

###tripletNNccca
prec_tptccca_va = [0.95, 0.945, 0.92, 0.88, 0.85 , 0.78, 0.66333333, 0.5714286, 0.525, 0.472, 0.42,
                   0.3783333, 0.2933333, 0.27,      
                   0.1801,       0.147,0.1109856,0.1049]
rec_tptccca_va  =  [0.0275, 0.08232, 0.10123, 0.191234, 0.2432, 0.28543,0.3123,
                   0.3522, 0.3543,  0.3878, 0.42234, 0.44632, 0.4712, 0.4953,
                   0.5632, 0.7124, 0.8921, 0.984]
####################################################

print("example", rec_dcca_va, prec_dcca_va)
print("audio-visual vegas dataset")
ax = plt.subplot(111)
plt.plot(rec_cca_av, prec_cca_av, label="CCA")
plt.plot(rec_dcca_av, prec_dcca_av, label="DCCA")
plt.plot(rec_ccca_av, prec_ccca_av, label="C-CCA")
plt.plot(rec_tptccca_av, prec_tptccca_av, label="TNN-C-CCA")
plt.plot(rec_sdcca_av, prec_sdcca_av, label="C-DCCA")
plt.plot(rec_ugach_av, prec_ugach_av, label="UGACH")
plt.plot(rec_ucal_av, prec_ucal_av, label="UCAL")
plt.plot(rec_acmr_av, prec_acmr_av, label="ACMR")
plt.plot(rec_agah_av, prec_agah_av, label="AGAH")
#plt.plot(rec_tptccca_av, prec_tptccca_av, label="TNN-C-CCA")

#red_patch = mpatches.Patch(color='red', label='The red data')
#plt.legend(handles=[red_patch])


plt.title('')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
leg = plt.legend(bbox_to_anchor=(0.65, 1),  ncol=1, mode=None, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.65)
plt.grid(True)
plt.savefig("/home/dhzeng/TNN_CCCA/PRC_image/PRC_VEGAS_AV.svg")
plt.show()
#################################################################################

print("visual-audio vegas dataset")
ax = plt.subplot(111)
ax = plt.subplot(111)
plt.plot(rec_cca_va, prec_cca_va, label="CCA")
plt.plot(rec_dcca_va, prec_dcca_va, label="DCCA")
plt.plot(rec_ccca_va, prec_ccca_va, label="C-CCA")
plt.plot(rec_tptccca_va, prec_tptccca_va, label="TNN-C-CCA")
plt.plot(rec_sdcca_va, prec_sdcca_va, label="C-DCCA")
plt.plot(rec_ugach_va, prec_ugach_va, label="UGACH")
plt.plot(rec_ucal_va, prec_ucal_va, label="UCAL")
plt.plot(rec_acmr_va, prec_acmr_va, label="ACMR")
plt.plot(rec_agah_va, prec_agah_va, label="AGAH")
#plt.plot(rec_tptccca_va, prec_tptccca_va, label="TNN-C-CCA")

plt.title('')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
leg = plt.legend(bbox_to_anchor=(0.65, 1), ncol=1, mode=None, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.65)
plt.grid(True)
plt.savefig("/home/dhzeng/TNN_CCCA/PRC_image/PRC_VEGAS_VA.svg")
plt.show()