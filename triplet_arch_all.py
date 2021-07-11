import numpy as np
import os

import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D, concatenate, Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
import keras

from itertools import permutations
#import seaborn as sns

import h5py
from myconfig import Myconfig
from keras.utils import to_categorical
import tensorflow, keras
import utils
from tensorflow.python.client import device_lib
# use the following code to let tensorflow only allocate memory as needed
#cfg = tensorflow.ConfigProto()
#cfg.gpu_options.allow_growth = True
#keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=cfg))

def triplet_losses(y_true, y_pred, margin=0.5):
    print("the margin value is like:".format(margin))
    total_lenght = y_pred.shape.as_list()[-1]

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + margin
    print("basic_loss is like+++++++++++++++++++++++++++++++++",basic_loss)
    print(basic_loss.shape)
    loss = K.maximum(basic_loss, 0.0)
    return loss
    
def triplet_loss(y_true, y_pred, margin=0.5):
    print("the margin value is like:".format(margin))
    total_lenght = y_pred.shape.as_list()[-1]

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = pairwise_cosine_sim(anchor, positive)

    # distance between the anchor and the negative
    neg_dist = pairwise_cosine_sim(anchor, negative)

    # compute loss
    basic_loss = pos_dist - neg_dist + margin
    print("basic_loss is like+++++++++++++++++++++++++++++++++",basic_loss)
    print(basic_loss.shape)
    loss = K.maximum(basic_loss, 0.0)
    return loss
    

def pairwise_cosine_sim(a, b):
    similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)
    # Only necessary if vectors are not normalized
    similarity /= tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1)
    # If you prefer the distance measure
    distance = 1 - similarity
    return distance
    
def batch_all_triplet_loss(y_true, y_pred, margin=0.5):
    total_lenght = y_pred.shape.as_list()[-1]
    Anchor = y_pred[:,0:int(total_lenght*1/3)]
    Positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    Negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    anchor_positive_dist  = pairwise_cosine_sim(Anchor, Positive)
    anchor_negative_dist  = pairwise_cosine_sim(Anchor, Negative)

    anchor_positive_dist = tf.linalg.tensor_diag_part(anchor_positive_dist)
    anchor_negative_dist = tf.linalg.tensor_diag_part(anchor_negative_dist)

    triplet_loss = tf.maximum(anchor_positive_dist - anchor_negative_dist + margin, 0.0)
##    triplet_loss = tf.squeeze(triplet_loss)

##    l_left_shift = tf.concat((triplet_loss[1:], [0]), axis=0)
##    mask_left_shift = tf.not_equal(triplet_loss - l_left_shift, 0)
##    mask = tf.concat(([True], mask_left_shift[:-1]), axis=0)
##    triplet_loss = tf.boolean_mask(triplet_loss, mask)

    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)

    triplet_loss = tf.reduce_sum(triplet_loss)
    triplet_loss = tf.c1ast(triplet_loss, tf.float32)
    triplet_loss = triplet_loss/(num_positive_triplets+1e-16)
    return triplet_loss

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

def myGenerator(config, train_audio, train_rgb, train_lab, batch_num=400):
    train_data_xy = [train_audio, train_rgb, train_lab]
    data_idx_lists = range(len(train_audio))
    batchs_data_idxs = np.array_split(data_idx_lists, batch_num)
    print("\nbatch number={}; batch size={}".format(len(batchs_data_idxs), len(batchs_data_idxs[0])))
    i = 0
    while True:
        for batch_idxs in batchs_data_idxs:
            print(batch_idxs)
            batch_train_audio = train_audio[batch_idxs]
            print(batch_train_audio.shape)
            batch_train_rgb = train_rgb[batch_idxs]
            batch_train_lab = train_lab[batch_idxs]

            print("batch-{} ".format(i))
            batch_triplet_list, Lab = [], []
            Anchor_list, Positive_list, Negative_list = [], [], []
            Anchor_x, Positive_x, Negative_x, Lab_x = np.empty((0, config.input), np.float32), np.empty(
                (0, config.input), np.float32), np.empty((0, config.input), np.float32), np.empty((0,), np.float32)
            for data_class in sorted(set(batch_train_lab)):
                ### same idxs
                same_class_idx = np.where(batch_train_lab == data_class)[0]
                ### diff idxs
                diff_class_idx = np.where(batch_train_lab != data_class)[0]
                A_P_pairs = list(permutations(same_class_idx, 2))
                Neg_idx = list(diff_class_idx)
                # print("same_class sample number is {}, neg idx number is {}, A_P_pairs {}".format(len(same_class_idx), len(diff_class_idx), len(A_P_pairs)))

                for ap in A_P_pairs:
                    Anchor = np.array(batch_train_audio[ap[0]], dtype=np.float32)
                    Positive = batch_train_rgb[ap[1]]
                    for Neg_index in Neg_idx:
                        Negative = batch_train_rgb[Neg_index]
                        Anchor_x = np.concatenate((Anchor_x, [Anchor]))
                        Positive_x = np.concatenate((Positive_x, [Positive]))
                        Negative_x = np.concatenate((Negative_x, [Negative]))
                        Lab_x = np.concatenate((Lab_x, [data_class]))
            ##                    print(Anchor_x.shape, Positive_x.shape,Negative_x.shape, Lab_x.shape)
            i += 1
            Lab_x = to_categorical(Lab_x)
            print('Batch valid triplets shape:', Anchor_x.shape, Positive_x.shape, Negative_x.shape, Lab_x.shape)
            yield [Anchor_x, Positive_x,
                   Negative_x], Lab_x  # np.array([Anchor_list, Positive_list, Negative_list]), Lab

def cross_fold_generate(path):
    file_list = os.listdir(path)

    for i in range(5):
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

config = Myconfig()

anchor_input = Input((10,))
print("anchor input:", anchor_input)
positive_input = Input((10,))
print("positive input:", positive_input)
negative_input = Input((10,))
print("negative input:", negative_input)
## Neural Networks
Anchor_DNN = create_network(10)
Shared_DNN = create_base_network(10)
encoded_anchor = Shared_DNN(anchor_input)  # Shared_DNN(anchor_input)
encoded_positive = Shared_DNN(positive_input)
encoded_negative = Shared_DNN(negative_input)

merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
Anchor_branch = Model(inputs=anchor_input, outputs=encoded_anchor)
Pos_neg_branch = Model(inputs=positive_input, outputs=encoded_positive)
## Compile
adam_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss=triplet_losses, optimizer=adam_optim)  # batch_hard_triplet_loss
print(model.summary())
# model.fit([A, B, B], C, batch_size=10, epochs=1)
path = "/home/dhzeng/AVIDEO/iEmbedding/ccca_vgg_inception10_02/"
i=0
metric_lst = []
for data_xy in cross_fold_generate(path):
    train_audio, train_rgb, train_lab, test_audio, test_rgb, test_lab = data_xy
    print("current fold as test; the train and test infomation:\n", \
          train_audio.shape, train_rgb.shape, train_lab.shape, \
          test_audio.shape, test_rgb.shape, test_lab.shape)
    model.fit_generator(myGenerator(config, train_audio, train_rgb, \
                                    train_lab, 1000), steps_per_epoch=400, nb_epoch=2)


    model.save_weights('triplet_batchs.hdf5')
    Anchor_branch.save_weights("Aencoded_anchor"+str(i)+".hdf5")
    Pos_neg_branch.save_weights("Aencoded_pos_neg"+str(i)+".hdf5")

    trained_anchor = Model(inputs=anchor_input, outputs=encoded_anchor)
    trained_pos_neg = Model(inputs=positive_input, outputs=encoded_positive)

    trained_anchor.load_weights("Aencoded_anchor"+str(i)+".hdf5")
    trained_pos_neg.load_weights("Aencoded_pos_neg"+str(i)+".hdf5")

    test_audio_x = trained_anchor.predict(test_audio)
    test_rgb_x = trained_pos_neg.predict(test_rgb)
    fds = "_{0:02d}s.h5".format(i)
    np.save("/home/dhzeng/tnn_c_cca/embedding/Accca_test_audio"+str(fds)+".npy", test_audio)
    np.save("/home/dhzeng/tnn_c_cca/embedding/Accca_test_visual"+str(fds)+".npy", test_rgb)
    
    np.save("/home/dhzeng/tnn_c_cca/embedding/Atnn_ccca_test_audio"+str(fds)+".npy", test_audio_x)
    np.save("/home/dhzeng/tnn_c_cca/embedding/Atnn_test_visual"+str(fds)+".npy", test_rgb_x)
    np.save("/home/dhzeng/tnn_c_cca/embedding/Atest_lab"+str(fds)+".npy", test_lab)

    print("The output of test:", test_audio.shape, test_rgb.shape)
    views = test_audio_x, test_rgb_x
    corr_matrix = utils.corr_compute(views, tag="cosine")
    try:
        np.save("corr"+str(i)+".npy", corr_matrix)
    except:
        print("error~~~~~~~~~~~~~~~")
    result_list = utils.metric(corr_matrix, test_lab)
    print(result_list)
    mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2, mean_prec_x, mean_rec_x, mean_prec_y, mean_rec_y \
                   = result_list
    metric_lst.append([mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2])
    i+=1

metric_matrix   = np.matrix(np.array(metric_lst)).mean(0)

result = "(Audio-Visual): map={0}, MRR1={1}\n(Visual-Audio): map={2}, MRR1={3}".\
             format(metric_matrix.item(0),metric_matrix.item(2), metric_matrix.item(1), metric_matrix.item(3))
print(result)
np.save("result.py", result)

