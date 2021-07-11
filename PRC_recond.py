#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:27:15 2019

@author: dhzeng
"""



###############################################################################
###############################################################################
#import _pickle as cPickle
import cPickle

kk=0
prec_ugach_av_x = []
rec_ugach_av_x = []
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
## MV-10K dataset.
#print("audio-visual mv10k dataset")
#ax = plt.subplot(111)
#output_num = [50, 100, 200, 300, 400, 500, 600, 800, 900, 1000, 1200, 1500, 1800, 2000]
#
### cca
#prec_cca_av = [0.08, 0.115,0.1045,0.107666667, 0.10775,0.109,0.109,0.108857143, 0.109,0.109555556, 0.1, 0.11, 0.106, 0.10]
##[0,  0.22373052, 0.21510264, 0.21834174, 
#              # 0.21488853, 0.21351534, 0.20791041, 0.20530959, 0.20358026, 0.20188309, 0.20188309, 0.200232,
#              # 0.1930237, 0.1920237, 0.19188513]
#rec_cca_av  = [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726,0.31196581, 0.35042735, 0.3974359, 0.44444444, 0.52136752, 0.61965812,0.78205128, 1.]#[0.00813008, 0.05691057, 0.11382114, 0.15447154, 0.17886179, 0.2195122, 0.27642276, 0.31707317,
#               #0.3495935,  0.42276423, 0.43089431, 0.55284553, 0.68292683, 1.]
#
### dcca
#prec_dcca_av = [0.16,       0.15,       0.125,      0.14,       0.13,       0.122,
# 0.13166667, 0.13857143, 0.13375,    0.13444444, 0.131 ,     0.12583333,
# 0.104,      0.10]#[0,  0.22042459, 0.2127434, 0.21119996, 0.21080105, 0.21051462, 0.210007021, 0.20937441, 0.20619967, 0.2008174597, 0.200174597, 0.1950756, 0.19370756,0.19270756, 0.19188513]
#rec_dcca_av  = [0.03145478, 0.0576671,  0.10747051, 0.16644823, 0.21494102, 0.26605505, 0.31454784, 0.35517693,
#                0.40629096, 0.46002621, 0.50327654, 0.60419397, 0.74442988, 1.        ]
#
### c-cca
#prec_ccca_av = [0.2160459, 0.2207434, 0.218336, 0.21400105, 0.201501462, 0.18307021, 0.1703375, 0.15144444, 0.137, 
#                0.13083333,  0.114,      0.110, 0.104,      0.10]
#rec_ccca_av  =   [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726,0.31196581, 0.35042735,
#                  0.3974359,  0.44444444, 0.52136752, 0.61965812, 0.78205128, 1.        ]
#
### s-dcca
#prec_sdcca_av = [0.2180459, 0.224434, 0.22796, 
#                0.22680105, 0.2231462, 0.198087021, 0.1837441, 0.16919967, 0.15174597, 
#                0.138756, 0.130756, 0.12188513, 0.104,      0.10]
#rec_sdcca_av  = [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726,0.31196581, 0.35042735,
#                  0.3974359,  0.44444444, 0.52136752, 0.61965812, 0.78205128, 1.        ]
#
### tripletNNccca
#prec_tptccca_av = [ 0.2190459, 0.22927434, 0.22996, 
#                0.2280105, 0.2259351462, 0.20587021, 0.187441, 0.172919967, 0.16018174597, 
#                0.1559756, 0.14399756, 0.13188513, 0.104,      0.10]
#rec_tptccca_av  = [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726,0.31196581, 0.35042735,
#                  0.3974359,  0.44444444, 0.52136752, 0.61965812, 0.78205128, 1.        ]
#
#
#
#plt.plot(rec_cca_av, prec_cca_av, label="audio_visual_cca_prc")
#plt.plot(rec_dcca_av, prec_dcca_av, label="audio_visual_dcca_prc")
#plt.plot(rec_ccca_av, prec_ccca_av, label="audio_visual_ccca_prc")
#plt.plot(rec_sdcca_av, prec_sdcca_av, label="audio_visual_sdcca_prc")
#plt.plot(rec_ugach_av_x, prec_ugach_av_x, label="audio_visual_ugach_prc")
#plt.plot(rec_tptccca_av, prec_tptccca_av, label="audio_visual_cccatriplet_prc")
#
#plt.title('The Precision-Recall Curve')
#plt.xlabel('Recall-axis', fontsize=14)
#plt.ylabel('Precision-axis', fontsize=14)
#leg = plt.legend(loc=8, ncol=2, mode="expand", shadow=True, fancybox=True)
#leg.get_frame().set_alpha(0.3)
#plt.grid(True)
#plt.show()

print("audio-visual mv10k dataset")
ax = plt.subplot(111)
output_num = [0, 50, 100, 200, 300, 400, 500, 600, 800, 900, 1000, 1200, 1500, 1800, 2000]

## cca
prec_cca_av = [0.22373052, 0.21510264, 0.21834174, 0.21488853, 0.21351534, 0.20791041, 0.20530959, 0.20358026, 0.20188309, 0.20188309, 0.200232,
               0.1930237, 0.1920237, 0.19188513]
rec_cca_av  =  [0.03145478, 0.0576671,  0.10747051, 0.16644823, 0.21494102, 0.26605505, 0.31454784, 0.35517693,
                0.40629096, 0.46002621, 0.50327654, 0.60419397, 0.74442988, 1.        ]

## dcca
prec_dcca_av = [ 0.22042459, 0.2127434, 0.21119996, 
                0.21080105, 0.21051462, 0.210007021, 0.20937441, 0.20619967, 0.2008174597, 0.200174597, 
                0.1950756, 0.19370756,0.19270756, 0.19188513]
rec_dcca_av  =  [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
                 0.3974359,  0.44444444, 0.52136752, 0.61965812, 0.78205128, 1.        ] 


## c-cca
prec_ccca_av = [ 0.2360459, 0.2407434, 0.240336, 0.2400105, 0.238501462, 0.23607021, 0.22837441, 0.221719967, 
                0.2174597, 0.21019967, 0.201174597, 0.1920756,0.1910756, 0.19188513]
rec_ccca_av  = [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
                0.3974359,  0.44444444, 0.52136752, 0.61965812,0.78205128, 1.        ]

## s-dcca
prec_sdcca_av = [0.2380459, 0.242434, 0.24296, 0.24180105, 0.2401462, 0.239087021, 0.2337441, 0.22919967, 
                 0.223174597, 0.21919967, 0.206174597, 0.1970756, 0.1930756, 0.19188513]
rec_sdcca_av  = [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
                0.3974359,  0.44444444, 0.52136752, 0.61965812,0.78205128, 1.        ]

## tripletNNccca
prec_tptccca_av = [ 0.2390459, 0.2427434, 0.242996, 0.24280105, 0.242351462, 0.241387021, 0.23837441, 0.231919967, 
                   0.228174597, 0.22019967, 0.210174597, 0.198799756, 0.1959756, 0.19188513]
rec_tptccca_av  = [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
                0.3974359,  0.44444444, 0.52136752, 0.61965812,0.78205128, 1.        ]



plt.plot(rec_cca_av, prec_cca_av, label="audio_visual_cca_prc")
plt.plot(rec_dcca_av, prec_dcca_av, label="audio_visual_dcca_prc")
plt.plot(rec_ccca_av, prec_ccca_av, label="audio_visual_ccca_prc")
plt.plot(rec_sdcca_av, prec_sdcca_av, label="audio_visual_sdcca_prc")
plt.plot(rec_ugach_av_x, prec_ugach_av_x, label="audio_visual_ugach_prc")

plt.plot(rec_tptccca_av, prec_tptccca_av, label="audio_visual_cccatriplet_prc")

plt.title('The Precision-Recall Curve')
plt.xlabel('Recall-axis', fontsize=14)
plt.ylabel('Precision-axis', fontsize=14)
leg = plt.legend(loc=3, ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.grid(True)
plt.show()

###################################################
### cca
prec_cca_va = [  0.22139831, 0.21997729, 0.2146568,
               0.21250659, 0.21243878, 0.20781722, 0.20561445, 0.20436282,0.20136282,
               0.19959329, 0.19959329, 0.19188513]
rec_cca_va  = [0.03145478, 0.0576671,  0.10747051, 0.16644823, 0.21494102, 0.26605505,
 0.31454784, 0.40629096, 0.50327654, 0.60419397, 0.74442988, 1.        ] 

### dcca
prec_dcca_va = [ 0.2192459, 0.219435042, 0.2131111, 
                0.209961054, 0.2091462, 0.2080218, 0.20530441, 0.20299657,0.20000000, 
                0.19359329, 0.19259329, 0.19188513]
rec_dcca_va  = [0.03145478, 0.0576671,  0.10747051, 0.16644823, 0.21494102, 0.26605505,
 0.31454784, 0.40629096, 0.50327654, 0.60419397, 0.74442988, 1.        ] 

### c-cca
prec_ccca_va = [ 0.23042459, 0.231435042, 0.23299962, 
                0.230801054, 0.228351462, 0.223870218, 0.22199137441, 0.2179657, 0.21174597,  0.2074597, 0.20274597, 
                0.2000949329, 0.19929329, 0.19188513]
rec_ccca_va  = [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
               0.3974359,  0.44444444, 0.52136752, 0.6196581, 0.78205128, 1.        ]

### s-dcca
prec_sdcca_va = [ 0.233049, 0.235435042, 0.2339962, 0.2320801054, 0.230151462, 0.228070218, 0.2237441, 0.2190199657, 
                 0.21674597, 0.20974597, 0.20674597, 0.2011999329, 0.19979329, 0.19198513]
rec_sdcca_va  = [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
               0.3974359,  0.44444444, 0.52136752, 0.6196581, 0.78205128, 1.        ]

###tripletNNccca
prec_tptccca_va = [ 0.23503592, 0.2361436755, 0.2350896421, 0.2342459, 0.2310435042, 0.230099962, 0.2260001054, 
                   0.22081462, 0.2180218, 0.21474597, 0.2074597, 0.2020000756,  0.19997756, 0.19188513]
rec_tptccca_va  = [0.02564103, 0.05982906, 0.12393162, 0.15384615, 0.20512821, 0.26495726, 0.31196581, 0.35042735,
               0.3974359,  0.44444444, 0.52136752, 0.6196581, 0.78205128, 1.        ]

####################################################
ax = plt.subplot(111)
plt.plot(rec_cca_va, prec_cca_va, label="visual_audio_cca_prc")
plt.plot(rec_dcca_va, prec_dcca_va, label="visual_audio_dcca_prc")
plt.plot(rec_ccca_va, prec_ccca_va, label="visual_audio_ccca_prc")
plt.plot(rec_sdcca_va, prec_sdcca_va, label="visual_audio_sdcca_prc")
plt.plot(rec_ugach_va_x, prec_ugach_va_x, label="visual_audio_ugach_prc")
plt.plot(rec_tptccca_va, prec_tptccca_va, label="visual_audio_cccatriplet_prc")

plt.title('The Precision-Recall Curve ')
plt.xlabel('Recall-axis', fontsize=14)
plt.ylabel('Precision-axis', fontsize=14)
leg = plt.legend(loc="best", ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.grid(True)
plt.show()

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
####################################################################
prec_ugach_av_x = []
rec_ugach_av_x = []
##################################################################################
##################################################################################

prc_mv10k_dicts={}
prc_mv10k_dicts['output_num']=output_num

prc_mv10k_dicts['cca']      = [prec_cca_av, rec_cca_av],         [prec_dcca_va, rec_dcca_va]
prc_mv10k_dicts['ccca']     = [prec_ccca_av, rec_ccca_av],       [prec_ccca_va, rec_ccca_va]
prc_mv10k_dicts['sdcca']    = [prec_sdcca_av, rec_sdcca_av],     [prec_sdcca_va, rec_sdcca_va]
prc_mv10k_dicts['tnn_ccca'] = [prec_tptccca_av, rec_tptccca_av], [prec_tptccca_va, rec_tptccca_va]
prc_mv10k_dicts["ugach"] = [prec_ugach_av_x, rec_ugach_av_x], [prec_ugach_va_x, rec_ugach_va_x]

with open("results/prc_mv10k_dicts.cpk", "wb") as fb:
    cPickle.dump(prc_mv10k_dicts, fb)
