from utils import corr_compute, metric
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D, concatenate, Activation, Dropout
from keras.models import Model, Sequential
import os
import h5py
import numpy as np

def cross_fold_generate(path):
    file_list = os.listdir(path)

    for i in range(1):
        # train
        train_emb = "train_{0:02d}s".format(i)
        ftr = h5py.File(path + train_emb + ".h5", 'r')
        train_audio = ftr["featN1"]
        train_rgb = ftr["featN2"]
        train_lab = ftr["testLs"]
        train_lab = np.asarray([int(labb) for labb in train_lab])

        # test
        test_emb = "test_{0:02d}s".format(i)
        ff = h5py.File(path + test_emb + ".h5", 'r')
        test_audio, test_rgb = ff["featN1"], ff["featN2"]
        test_lab = ff["testLs"]
        print("train, test shape like {},{},{}; {},{},{}".format(train_audio.shape, train_rgb.shape, train_lab.shape,
                                                                 test_audio.shape, test_rgb.shape, test_lab.shape))
        print(type(train_audio), type(train_rgb), type(train_lab), type(test_audio), type(test_rgb), type(test_lab))
        yield np.array(train_audio, dtype=np.float32), np.array(train_rgb, dtype=np.float32), np.array(train_lab,
                                                                                                       dtype=np.float32), np.array(
            test_audio, dtype=np.float32), np.array(test_rgb, dtype=np.float32), np.array(test_lab, dtype=np.float32)


def create_base_network(in_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Dense(100, input_dim=in_dims))
    model.add(Activation("tanh"))
    model.add(Dropout(0.1))
    model.add(Dense(100, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(100, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(10, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Activation("sigmoid"))
    return model

def create_network(in_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Dense(100, input_dim=in_dims))
    model.add(Activation("tanh"))
    model.add(Dropout(0.1))
    model.add(Dense(100, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(100, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(10, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Activation("sigmoid"))
    return model

anchor_input = Input((10,))
print("anchor input:", anchor_input)
positive_input = Input((10,))
print("positive input:", positive_input)
negative_input = Input((10,))
print("negative input:", negative_input)
## Neural Networks
Anchor_DNN = create_network(10)
Shared_DNN = create_base_network(10)
encoded_anchor = Anchor_DNN(anchor_input)  # Shared_DNN(anchor_input)
encoded_positive = Shared_DNN(positive_input)
encoded_negative = Shared_DNN(negative_input)

path = "/home/dhzeng/AVIDEO/iEmbedding/ccca_vgg_inception10_02/"
for data_xy in cross_fold_generate(path):
    train_audio, train_rgb, train_lab, test_audio, test_rgb, test_lab = data_xy

    trained_anchor = Model(inputs=anchor_input, outputs=encoded_anchor)
    trained_pos_neg = Model(inputs=positive_input, outputs=encoded_positive)

    trained_anchor.load_weights('encoded_anchor.hdf5')
    trained_pos_neg.load_weights('encoded_pos_neg.hdf5')

    test_audio = trained_anchor.predict(test_audio)
    test_rgb = trained_pos_neg.predict(test_rgb)

    print("The output of test:", test_audio.shape, test_rgb.shape)
    views = test_audio, test_rgb
    corr_matrix = corr_compute(views, tag="cosine")
    try:
        np.save("corr.npy", corr_matrix)
    except:
        print("save corr error~~~~~~~~~~~~~~~")
    result_list = metric(corr_matrix, test_lab)
    print(result_list)

