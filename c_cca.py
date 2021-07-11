import numpy as np
import utils
import h5py
import os
from linear_cca import linear_c_cca
import itertools
from eval import corr_compute, metric, fx_calc_map_label

def ccca_emb(loadfile, output_size, beta):
	Reader = h5py.File(loadfile, 'r')
	audioTrain =  Reader['audio_train'][()]
	audioTest  =  Reader['audio_test'][()]

	rgbTrain   =  Reader['visual_train'][()]
	rgbTest    =  Reader['visual_test'][()]

	labelTrain =  Reader['lab_train'][()]
	labelTest  =  Reader['lab_test'][()]

	print("Linear CCA started!")
	w0, w1, m0, m1 = linear_c_cca(audioTrain, rgbTrain, labelTrain, False, output_size, beta)
	np.savez('Model_Cc_CCA.npz', w0=w0, w1=w1, m0=m0, m1=m1)
	print("Linear CCA ended!")

	data_num = len(audioTrain)
	audioTrain -= m0.reshape([1, -1]).repeat(data_num, axis=0)
	audioFeatTrain = np.dot(audioTrain, w0)
	rgbTrain -= m1.reshape([1, -1]).repeat(data_num, axis=0)
	rgbFeatTrain = np.dot(rgbTrain, w1)

	data_num = len(audioTest)
	audioTest -= m0.reshape([1, -1]).repeat(data_num, axis=0)
	audioFeatTest = np.dot(audioTest, w0)
	rgbTest -= m1.reshape([1, -1]).repeat(data_num, axis=0)
	rgbFeatTest = np.dot(rgbTest, w1)

	return  audioFeatTrain, rgbFeatTrain, labelTrain, audioFeatTest, rgbFeatTest, labelTest

import time
start = time.time()
audioFeatTrain, rgbFeatTrain, labelTrain, audioFeatTest, rgbFeatTest, labelTest = ccca_emb("Data/vegas_feature.h5", 10, 0.2)
# audioFeatTrain, rgbFeatTrain, labelTrain, audioFeatTest, rgbFeatTest, labelTest = ccca_emb("Data/ave/AVE_feature_update1_squence.h5", 15, 0.2)
av_map, va_map = fx_calc_map_label(audioFeatTest, rgbFeatTest, labelTest, k = 0, dist_method='COS'), \
    fx_calc_map_label(rgbFeatTest, audioFeatTest, labelTest, k = 0, dist_method='COS')
average_map = (av_map+va_map)/2.0
print(("audio_visual map = {}\nvisual_audio map = {}\naverage = {}").format(av_map, va_map, average_map))
end = time.time()
print("You have taken times-2:", end - start)
# views, testLs = dcca_emb("Data/ave/AVE_feature_update1_squence.h5", output_size, batch_size, Applied_CCA)
# corr_matrix = corr_compute(audioFeatTest, rgbFeatTest, "cosine")
# result_list = metric(corr_matrix, labelTest)
# print(result_list)

# print(audioFeatTrain.shape, rgbFeatTrain.shape, labelTrain.shape, audioFeatTest.shape, rgbFeatTest.shape, labelTest.shape)
# with h5py.File('embedding/c_cca_test_new.h5', 'w') as h:
#     h.create_dataset("audio_test", data=audioFeatTest)
#     h.create_dataset("visual_test", data=rgbFeatTest)
#     h.create_dataset("lab_test", data=labelTest)
#     h.close()

# with h5py.File('embedding/c_cca_train_new.h5', 'w') as f:
#     f.create_dataset("audio_train", data=audioFeatTrain)
#     f.create_dataset("visual_train", data=rgbFeatTrain)
#     f.create_dataset("lab_train", data=labelTrain)
#     f.close()