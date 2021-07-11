#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:34:17 2019

@author: dhzeng
"""

import os, os.path
import numpy as np
import utils
from sklearn.decomposition import PCA
import seaborn as sns

dataPath = "/home/dhzeng/tnn_c_cca/Dataset/"
# audio
testpath   = os.path.join(dataPath, "audioTest00.npz")
trainpath  = os.path.join(dataPath, "audioTrain00.npz")
testAudio  = np.load(testpath)['arr_0']
trainAudio = np.load(trainpath)['arr_0']

# rgb
testpath  = os.path.join(dataPath, "rgbTest00.npz")
trainpath = os.path.join(dataPath, "rgbTrain00.npz")
testRGB   = np.load(testpath)['arr_0']
trainRGB  = np.load(trainpath)['arr_0']

# label
testpath  = os.path.join(dataPath, "labTest00.npz")
trainpath = os.path.join(dataPath, "labTrain00.npz")    
catTest   = np.load(testpath)['arr_0']
catTrain  = np.load(trainpath)['arr_0']

i=0
metric_lst = []

train_audio, train_rgb, train_lab, test_audio, test_rgb, test_lab = trainAudio, trainRGB, catTrain,  testAudio, testRGB, catTest
#X_TSNE = TSNE(n_components=2, perplexity=113.0,random_state=0)
X_TSNE = PCA(n_components=10)
test_audio_reduced = X_TSNE.fit_transform(test_audio)
test_rgb_reduced = X_TSNE.fit_transform(test_rgb)
views = test_audio_reduced, test_rgb_reduced
corr_matrix = utils.corr_compute(views, tag="cosine")

result_list = utils.metric(corr_matrix, test_lab)
print(result_list)