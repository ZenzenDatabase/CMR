#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:03:07 2019

@author: dhzeng
"""

import cPickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.spatial.distance import cdist

print("audio-visual mv10k dataset")
ax = plt.subplot(111)
output_num = [0, 50, 100, 200, 300, 400, 500, 600, 800, 900, 1000, 1200, 1500, 1800, 2000]

## cca
prec_cca_av = [0.22373052, 0.21510264, 0.21834174, 0.21488853, 0.21351534, 0.20791041, 0.20530959, 0.20358026, 0.20188309, 0.20188309, 0.200232,
               0.1930237, 0.1920237, 0.19188513]
rec_cca_av  =  [0.0083145478, 0.0576671,  0.10747051, 0.16644823, 0.21494102, 0.26605505, 0.31454784, 0.35517693,
                0.40629096, 0.46002621, 0.50327654, 0.60419397, 0.74442988, 1.        ]

## dcca
prec_dcca_av = [ 0.22042459, 0.2127434, 0.21119996, 
                0.21080105, 0.21051462, 0.210007021, 0.20937441, 0.20619967, 0.2008174597, 0.200174597, 
                0.1950756, 0.19370756,0.19270756, 0.19188513]
rec_dcca_av  =  [0.0082564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
                 0.3974359,  0.44444444, 0.52136752, 0.61965812, 0.78205128, 1.        ] 


## c-cca
prec_ccca_av = [ 0.2360459, 0.2407434, 0.240336, 0.2400105, 0.238501462, 0.23607021, 0.22837441, 0.221719967, 
                0.2174597, 0.21019967, 0.201174597, 0.1920756,0.1910756, 0.19188513]
rec_ccca_av  = [0.0082564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
                0.3974359,  0.44444444, 0.52136752, 0.61965812,0.78205128, 1.        ]

## s-dcca
prec_sdcca_av = [0.2380459, 0.242434, 0.24296, 0.24180105, 0.2401462, 0.239087021, 0.2337441, 0.22919967, 
                 0.223174597, 0.21919967, 0.206174597, 0.1970756, 0.1930756, 0.19188513]
rec_sdcca_av  = [0.0082564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
                0.3974359,  0.44444444, 0.52136752, 0.61965812,0.78205128, 1.        ]

## tripletNNccca
prec_tptccca_av = [ 0.2390459, 0.2427434, 0.242996, 0.24280105, 0.242351462, 0.241387021, 0.23837441, 0.231919967, 
                   0.228174597, 0.22019967, 0.210174597, 0.198799756, 0.1959756, 0.19188513]
rec_tptccca_av  = [0.0082564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
                0.3974359,  0.44444444, 0.52136752, 0.61965812,0.78205128, 1.        ]

prec_cca_va = [  0.22139831, 0.21997729, 0.2146568,
               0.21250659, 0.21243878, 0.20781722, 0.20561445, 0.20436282,0.20136282,
               0.19959329, 0.19959329, 0.19188513]
rec_cca_va  = [0.0083145478, 0.0576671,  0.10747051, 0.16644823, 0.21494102, 0.26605505,
 0.31454784, 0.40629096, 0.50327654, 0.60419397, 0.74442988, 1.        ] 

### dcca
prec_dcca_va = [ 0.2192459, 0.219435042, 0.2131111, 
                0.209961054, 0.2091462, 0.2080218, 0.20530441, 0.20299657,0.20000000, 
                0.19359329, 0.19259329, 0.19188513]
rec_dcca_va  = [0.0083145478, 0.0576671,  0.10747051, 0.16644823, 0.21494102, 0.26605505,
 0.31454784, 0.40629096, 0.50327654, 0.60419397, 0.74442988, 1.        ] 

### c-cca
prec_ccca_va = [ 0.23042459, 0.231435042, 0.23299962, 
                0.230801054, 0.228351462, 0.223870218, 0.22199137441, 0.2179657, 0.21174597,  0.2074597, 0.20274597, 
                0.2000949329, 0.19929329, 0.19188513]
rec_ccca_va  = [0.0082564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
               0.3974359,  0.44444444, 0.52136752, 0.6196581, 0.78205128, 1.        ]

### c-dcca
prec_sdcca_va = [ 0.233049, 0.235435042, 0.2339962, 0.2320801054, 0.230151462, 0.228070218, 0.2237441, 0.2190199657, 
                 0.21674597, 0.20974597, 0.20674597, 0.2011999329, 0.19979329, 0.19198513]
rec_sdcca_va  = [0.0082564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
               0.3974359,  0.44444444, 0.52136752, 0.6196581, 0.78205128, 1.        ]

###tripletNNccca
prec_tptccca_va = [ 0.24503592, 0.2391436755, 0.2370896421, 0.2362459, 0.2350435042, 0.232099962, 0.22650001054, 
                   0.22081462, 0.2180218, 0.21474597, 0.2074597, 0.2020000756,  0.19997756, 0.19188513]
rec_tptccca_va  = [0.0082564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
               0.3974359,  0.44444444, 0.52136752, 0.6196581, 0.78205128, 1.        ]
prec_ugach_av_x, rec_ugach_av_x = [], []
kk=0
with open("/home/dhzeng/TNN_CCCA/UGACH/Result/prca", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.split()
            prec_ugach_av_x.append(float(pre))
            rec_ugach_av_x.append(float(rec))
        kk+=1
        
prec_ugach_va_x = []
rec_ugach_va_x = []

kk=0
with open("/home/dhzeng/TNN_CCCA/UGACH/Result/prcb", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.split()
            prec_ugach_va_x.append(float(pre))
            rec_ugach_va_x.append(float(rec))
        kk+=1

#******************************ucal*****************************
prec_ucal_av = []
rec_ucal_av = []
kk=0
with open("/home/dhzeng/TNN_CCCA/ACMR/result/prc_av_ucal_data2.txt", "r") as f:
    for line in f.readlines():
        print(line.strip().split())
        if kk>0:
            pre, rec = line.strip().split()
            prec_ucal_av.append(float(pre))
            rec_ucal_av.append(float(rec))
        kk+=1
        
prec_ucal_va = []
rec_ucal_va = []
kk=0
with open("/home/dhzeng/TNN_CCCA/ACMR/result/prc_va_ucal_data2.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.strip().split()
            prec_ucal_va.append(float(pre))
            rec_ucal_va.append(float(rec))
        kk+=1
#*****************************acmr*****************************
prec_acmr_av_x = []
rec_acmr_av_x = []
kk=0
with open("/home/dhzeng/TNN_CCCA/ACMR/result/prc2_av_cl.txt", "r") as f:
    for line in f.readlines():
        print(line.strip().split())
        if kk>0:
            pre, rec = line.strip().split()
            prec_acmr_av_x.append(float(pre))
            rec_acmr_av_x.append(float(rec))
        kk+=1
        
prec_acmr_va_x = []
rec_acmr_va_x = []
kk=0
with open("/home/dhzeng/TNN_CCCA/ACMR/result/prc2_va_cl.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.strip().split()
            prec_acmr_va_x.append(float(pre))
            rec_acmr_va_x.append(float(rec))
        kk+=1
#******************************agah*****************************
prec_agah_av = []
rec_agah_av = []
kk=0
with open("/home/dhzeng/TNN_CCCA/AGAH/result/prc_av_data2.txt", "r") as f:
    for line in f.readlines():
        print(line.strip().split())
        if kk>0:
            pre, rec = line.strip().split()
            prec_agah_av.append(float(pre))
            rec_agah_av.append(float(rec))
        kk+=1
        
prec_agah_va = []
rec_agah_va = []
kk=0
with open("/home/dhzeng/TNN_CCCA/AGAH/result/prc_va_data2.txt", "r") as f:
    for line in f.readlines():
        if kk>0:
            pre, rec = line.strip().split()
            prec_agah_va.append(float(pre))
            rec_agah_va.append(float(rec))
        kk+=1

#################################################################################
        
print("audio-visual mv10k dataset")
ax = plt.subplot(111)
plt.plot(rec_cca_av, prec_cca_av, label="CCA")
plt.plot(rec_dcca_av, prec_dcca_av, label="DCCA")
plt.plot(rec_ccca_av, prec_ccca_av, label="C-CCA")
plt.plot(rec_tptccca_av, prec_tptccca_av, label="TNN-C-CCA")
plt.plot(rec_sdcca_av, prec_sdcca_av, label="C-DCCA")
plt.plot(rec_ugach_av_x, prec_ugach_av_x, label="UGACH")
plt.plot(rec_ucal_av, prec_ucal_av, label="UCAL")
plt.plot(rec_acmr_av_x, prec_acmr_av_x, label="ACMR")
plt.plot(rec_agah_av, prec_agah_av, label="AGAH")
#plt.plot(rec_tptccca_av, prec_tptccca_av, label="TNN-C-CCA")

plt.title('')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
leg = plt.legend(bbox_to_anchor=(0.65, 1),  ncol=1, mode=None, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.65)
plt.grid(True)
plt.savefig("/home/dhzeng/TNN_CCCA/PRC_image/PRC_MV10K_AV.svg")
plt.show()

###############################################################################
ax = plt.subplot(111)
plt.plot(rec_cca_va, prec_cca_va, label="CCA")
plt.plot(rec_dcca_va, prec_dcca_va, label="DCCA")
plt.plot(rec_ccca_va, prec_ccca_va, label="C-CCA")
plt.plot(rec_tptccca_va, prec_tptccca_va, label="TNN-C-CCA")
plt.plot(rec_sdcca_va, prec_sdcca_va, label="C-DCCA")
plt.plot(rec_ugach_va_x, prec_ugach_va_x, label="UGACH")
plt.plot(rec_ucal_va, prec_ucal_va, label="UCAL")
plt.plot(rec_acmr_va_x, prec_acmr_va_x, label="ACMR")
plt.plot(rec_agah_va, prec_agah_va, label="AGAH")
#plt.plot(rec_tptccca_va, prec_tptccca_va, label="visual_audio_cccatriplet_prc")

plt.title('')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
leg = plt.legend(bbox_to_anchor=(0.65, 1),  ncol=1, mode=None, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.65)
plt.grid(True)
plt.savefig("/home/dhzeng/TNN_CCCA/PRC_image/PRC_MV10K_VA.svg")
plt.show()
prec_ugach_av = []
rec_ugach_av = []