#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:20:42 2019

@author: dhzeng
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.spatial.distance import cdist


ax = plt.subplot(111)
componentx =  ['10', '20', '30', '40', '50']
cca_av    =  [0.3243, 0.3210, 0.3126, 0.2912, 0.2804]
ccca_av   = [0.6516, 0.6310, 0.6101, 0.5812, 0.5504]
sdcca_av  = [0.7034, 0.6822, 0.6534, 0.6234, 0.5865]
tripletNN_av = [0.7466, 0.7214, 0.6932, 0.6801, 0.6621]
plt.plot(componentx, cca_av, label="CCA")
plt.plot(componentx, ccca_av, label="C-CCA")
plt.plot(componentx, sdcca_av, label="S-DCCA")
plt.plot(componentx, tripletNN_av, label="TNN-C-CCA")

plt.title('')
plt.xlabel('Correlation component', fontsize=13)
plt.ylabel('MAP', fontsize=13)
leg = plt.legend(bbox_to_anchor=(0.95, 0.5), ncol=1, mode=None,  shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.65)
plt.grid(True)
plt.savefig("/home/dhzeng/TNN_CCCA/PRC_image/component_AV.svg")
plt.show()

ax = plt.subplot(111)
componenty = ['10', '20', '30', '40', '50']
cca_va    =  [0.3211, 0.3113, 0.3064, 0.2834, 0.2746]
ccca_va   = [0.6435, 0.6312, 0.6114, 0.5934, 0.55415]
sdcca_va  = [0.6927, 0.6745, 0.6511, 0.6274, 0.5932]
tripletNN_va = [0.7377, 0.7154, 0.6901, 0.6698, 0.6542]
plt.plot(componenty, cca_va, label="CCA")
plt.plot(componenty, ccca_va, label="C-CCA")
plt.plot(componenty, sdcca_va, label="S-DCCA")
plt.plot(componenty, tripletNN_va, label="TNN-C-CCA")

plt.title('')
plt.xlabel('Correlation component', fontsize=13)
plt.ylabel('MAP', fontsize=13)
leg = plt.legend(bbox_to_anchor=(0.95, 0.5), ncol=1, mode=None,  shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.65)
plt.grid(True)
plt.savefig("/home/dhzeng/TNN_CCCA/PRC_image/component_VA.svg")
plt.show()
