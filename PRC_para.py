#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:45:25 2019

@author: dhzeng
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.spatial.distance import cdist

 
#with open("models_prec_rec_dicts.cpk", "wb") as f:
#    cp.dump(models_prec_rec_dicts, f)
#    f.close()


ax = plt.subplot(111)
batch_num = [300, 350, 400, 450, 500, 550, 600, 650, 700, 800, 900]
t1 = [0.7164, 0.7279, 0.7285, 0.7322, 0.7298, 0.7302, 0.7326, 0.7320, 0.7286, 0.7045, 0.6984]
t2 = [0.7449, 0.7481, 0.7531, 0.7511, 0.7501, 0.7499, 0.7487, 0.7458, 0.7412, 0.6899, 0.6882]
t3 = [0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034]
t4 = [0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927]
plt.plot(batch_num, t1, label="TNN-C-CCA (av)")
plt.plot(batch_num, t2, label="TNN-C-CCA (va)")
plt.plot(batch_num, t3, label="C-DCCA (av)")
plt.plot(batch_num, t4, label="C-DCCA (va)")

plt.title('')
plt.xlabel('Batch number', fontsize=13)
plt.ylabel('MAP', fontsize=13)
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.grid(True)
plt.savefig("/home/dhzeng/TNN_CCCA/PRC_image/batchsize.svg")
plt.show()

ax = plt.subplot(111)
margin = np.arange(0.1, 1.2, 0.1)
t1=  [0.6865, 0.6982, 0.7245, 0.7320, 0.7326, 0.7242, 0.7242, 0.7212, 0.7304, 0.6996, 0.6868]
t2 = [0.6836, 0.6979, 0.7430, 0.7459, 0.7531, 0.7417, 0.7415, 0.7380, 0.7468, 0.6930, 0.6847]
t3 = [0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034, 0.7034]
t4 = [0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927, 0.6927]
plt.plot(margin, t1, label="TNN-C-CCA (av)")
plt.plot(margin, t2, label="TNN-C-CCA (va)")
plt.plot(margin, t3, label="C-DCCA (av)")
plt.plot(margin, t4, label="C-DCCA (va)")

plt.title('')
plt.xlabel('Margin', fontsize=13)
plt.ylabel('MAP', fontsize=13)
leg = plt.legend(loc=6, ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.2)
plt.grid(True)
plt.savefig("/home/dhzeng/TNN_CCCA/PRC_image/margin.svg")
plt.show()

