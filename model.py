import os
import numpy as np
import tensorflow as tf
from sklearn.cross_decomposition import CCA

from MultiCCA import *

import utils

def cca_emb(train_audio_rgb_label, test_audio_rgb_label, load_emb_path):#, output_size, beta):
    audioTrain, rgbTrain, _          = train_audio_rgb_label
    audioTest,  rgbTest,  labelTest  = test_audio_rgb_label
    cca = CCA(n_components=10)
    cca.fit(audioTrain, rgbTrain)
    emb_audio, emb_rgb = cca.transform(audioTest, rgbTest)
    return (emb_audio, emb_rgb), labelTest

def ccca_emb(train_audio_rgb_label, test_audio_rgb_label, testModel, load_emb_path, output_size, beta):
    audioTrain, rgbTrain, labelTrain = train_audio_rgb_label
    audioTest,  rgbTest,  labelTest  = test_audio_rgb_label
    if (testModel):
        model = np.load(os.path.join(load_emb_path, 'Model_C_CCA.npz'))
        w0 = model['w0']
        w1 = model['w1']
        m0 = model['m0']
        m1 = model['m1']
    else:
        print("Linear_CCA started!")
        w0, w1, m0, m1 = linear_cca(audioTrain, rgbTrain, labelTrain, testModel, output_size, beta)
        np.savez(os.path.join(load_emb_path, 'Model_C_CCA.npz'), w0=w0, w1=w1, m0=m0, m1=m1)
        print("Linear_CCA ended!")
    data_num = len(audioTest)
    audioTest -= m0.reshape([1, -1]).repeat(data_num, axis=0)
    audioFeatTest = np.dot(audioTest, w0)
    rgbTest -= m1.reshape([1, -1]).repeat(data_num, axis=0)
    rgbFeatTest = np.dot(rgbTest, w1)
    return (audioFeatTest, rgbFeatTest), labelTest
    
def sdcca_emb(train_audio_rgb_label, test_audio_rgb_label, load_emb_path, output_size, beta):
    audioTrain, rgbTrain, labelTrain = train_audio_rgb_label
    audioTest,  rgbTest,  labelTest  = test_audio_rgb_label
    trainLabel = np.expand_dims(labelTrain, axis=1)
    testLabel  = np.expand_dims(labelTest, axis=1)
    train_size = len(audioTrain)
    dataAudio, dataRgb, dataLab = [audioTrain[:train_size], audioTrain[:50], audioTest], \
                                  [rgbTrain[:train_size],   rgbTrain[:50],   rgbTest], \
                                  [trainLabel[:train_size], trainLabel[:50], testLabel]
    model = train_model(dataAudio, dataRgb, dataLab, output_size, beta)
    trainDccaModel = True
    featNew = test_model(dataAudio, dataRgb, dataLab, trainDccaModel, output_size, beta)
    return (featNew[2][0], featNew[2][1]), testLabel

def dcca_emb(train_audio_rgb_label, test_audio_rgb_label, load_emb_path, output_size, beta):
    audioTrain, rgbTrain, labelTrain = train_audio_rgb_label
    audioTest,  rgbTest,  labelTest  = test_audio_rgb_label
    trainLabel = np.expand_dims(labelTrain, axis=1)
    testLabel  = np.expand_dims(labelTest, axis=1)
    train_size = len(audioTrain)
    dataAudio, dataRgb, dataLab = [audioTrain[:train_size], audioTrain[:50], audioTest],\
                                  [rgbTrain[:train_size],   rgbTrain[:50],   rgbTest], \
                                  [trainLabel[:train_size], trainLabel[:50], testLabel]
    model   = training_model(dataAudio, dataRgb, output_size, beta)
    featN = testing_model(dataAudio, dataRgb, output_size, beta)
    return (featN[2][0], featN[2][1]), testLabel

def seg_lstm_cca_emb(train_audio_rgb_label, test_audio_rgb_label, load_emb_path, output_size, beta):
    audioTrain, rgbTrain, labelTrain = train_audio_rgb_label
    audioTest,  rgbTest,  labelTest  = test_audio_rgb_label
    trainLabel = np.expand_dims(labelTrain, axis=1)
    testLabel  = np.expand_dims(labelTest, axis=1)
    train_size = len(audioTrain)
    dataAudio, dataRgb, dataLab = [audioTrain[:train_size], audioTrain[:50], audioTest], \
                                  [rgbTrain[:train_size],   rgbTrain[:50],   rgbTest], \
                                  [trainLabel[:train_size], trainLabel[:50], testLabel]
    trainDccaModels = True
    if (trainDccaModels):
        model = trained_model(dataAudio, dataRgb, dataLab, output_size, beta)
    featNew = tested_model(dataAudio, dataRgb, dataLab, trainDccaModels, output_size, beta)
    return (featN[2][0], featN[2][1]), testLabel

def sdcca_embedding(train_audio_rgb_label, test_audio_rgb_label, load_emb_path, output_size, beta):
    audioTrain, rgbTrain, labelTrain = train_audio_rgb_label
    audioTest,  rgbTest,  labelTest  = test_audio_rgb_label
    trainLabel = np.expand_dims(labelTrain, axis=1)
    testLabel  = np.expand_dims(labelTest, axis=1)
    train_size = len(audioTrain)
    dataAudio, dataRgb, dataLab = [audioTrain[:train_size], audioTrain[:50], audioTest], \
                                  [rgbTrain[:train_size],   rgbTrain[:50],   rgbTest], \
                                  [trainLabel[:train_size], trainLabel[:50], testLabel]

    outdim_size = 10
    learning_rate = 1e-3
    reg_par = 1e-5
    use_all_singular_values = False
    beta = 0.2
    input_shape1 = 128
    input_shape2 = 1024
    layer_sizes1 = [20, 20, outdim_size]
    layer_sizes2 = [20, 20, outdim_size]
    model = creat_model_dcca(layer_sizes1, layer_sizes2, input_shape1, input_shape2, learning_rate, reg_par, outdim_size, use_all_singular_values, beta)
    print(model.summary())
    model = train_model_dcca(model, dataAudio, dataRgb, dataLab, outdim_size, beta)
    featNew = test_model_dcca(model, dataAudio, dataRgb, dataLab, outdim_size, beta, True)
##    print("size of featNew[2][0] numbers: %f" % sys.getsizeof(featNew[2][0]))
##    print("size of featNew[2][1] numbers: %f" % sys.getsizeof(featNew[2][1]))
##    print("size of testLabel numbers: %f" % sys.getsizeof(testLabel))
    return (featNew[2][0], featNew[2][1]), testLabel
