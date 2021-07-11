#!/usr/bin/env python
import numpy as np
import random
import matplotlib.patheffects as PathEffects

from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate, Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import os
import pickle
import matplotlib.pyplot as plt

from itertools import permutations
import seaborn as sns

from keras.datasets import mnist
from sklearn.manifold import TSNE
import  tensorflow as tf
from sklearn.svm import SVC
from myconfig import Myconfig
from sklearn.preprocessing import LabelBinarizer

import h5py
import sys
import utils

def pairwise_cosine_sim(a, b):
    similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)
    # Only necessary if vectors are not normalized
    similarity /= tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1)
    # If you prefer the distance measure
    distance = 1 - similarity
    return distance

def custom_loss(labels):
    def loss(y_true, y_pred, alpha = 0.4):
        print('labelllllllllll',labels)
        total_lenght = y_pred.shape.as_list()[-1]

        view1 = y_pred[:,0:int(total_lenght*1/2)]
        view2 = y_pred[:,int(total_lenght*1/2):int(total_lenght*2/2)]
     
        Anchor, Positive, Negative = tf.constant([]), tf.constant([]), tf.constant([])
        for i in labels:
            same_class_idx = np.where((y_true == i))[0]
            diff_class_idx = np.where((y_true!= i))[0]
            A_P_pairs = list(permutations(same_class_idx,2))
            Neg_idx = list(diff_class_idx)
            ## Hard Positive pairs
            for ap in A_P_pairs:
                anchor = view1[ap[0]]
                positive = view1[ap[1]]
                for neg in Neg_idx:
                    negative = view2[neg]
                    Anchor = tf.concat([Anchor, anchor], 0)
                    #Anchor.append(anchor)
                    Positive = tf.concat([Positive, positive], 0)
                    #Positive.append(positive)
                    Negative = tf.concat([Negative, negative], 0)
                    #Negative.append(negative)
        #Anchor   = np.array(Anchor)
        #Positive = np.array(Positive)
        #Negative = np.array(Negative)
        print(Anchor, Positive,'------------------------')
        # distance between the anchor and the positive
        anchor_positive_dist  = pairwise_cosine_sim(Anchor, Positive)
        print('anchor_positive_dist',anchor_positive_dist)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        # distance between the anchor and the negative
        anchor_negative_dist = pairwise_cosine_sim(Anchor, Negative)
        hardest_negative_dist =tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
        triplet_loss = tf.squeeze(triplet_loss)

        #triplet_loss,y = tf.unique(triplet_loss)
        l_left_shift = tf.concat((triplet_loss[1:], [0]), axis=0)
        mask_left_shift = tf.not_equal(triplet_loss - l_left_shift, 0)
        mask = tf.concat(([True], mask_left_shift[:-1]), axis=0)
        triplet_loss = tf.boolean_mask(triplet_loss, mask)

        triplet_loss = tf.reduce_mean(triplet_loss)
        return triplet_loss
    return loss

    
def create_base_network(in_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Dense(300, input_dim=in_dims))
    model.add(Activation("tanh"))
    model.add(Dropout(0.1))
    model.add(Dense(300, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(300, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(10, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Activation("sigmoid"))
    # model.add(Dense(600))

    return model

def cross_fold_generate(config):
    path = config.path
    files_list = os.listdir(path)
    for i in range(1):# len(files_list)
        new_files_list = files_list[:i]+files_list[i+1:]
        ## read out the file
        ff = h5py.File(path+files_list[i], 'r')
        test_audio, test_rgb = ff["featN1"], ff["featN2"]
        test_lab = ff["testLs"]

        ## train files
        train_audio, train_rgb, train_lab = np.empty((0, config.input), np.float32), np.empty((0, config.input), np.float32), np.empty((0,), np.float32)
        for ft in new_files_list:
            ftr = h5py.File(path+ft, 'r')
            ft_audio = ftr["featN1"]
            ft_rgb   = ftr["featN2"]
            ft_lab = ftr["testLs"]
            ft_lab = np.asarray([int(labb) for labb in ft_lab])
            train_audio = np.concatenate((train_audio, ft_audio))
            train_rgb   = np.concatenate((train_rgb, ft_rgb))
            train_lab   = np.concatenate((train_lab, ft_lab))
        yield train_audio, train_rgb, train_lab, test_audio, test_rgb, test_lab

config = Myconfig()
adam_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

view1_input = Input((10,))
view2_input = Input((10, ))

# Shared embedding layer for positive and negative items
Shared_DNN = create_base_network(10)

encoded_view1 = Shared_DNN(view1_input)
encoded_view2 = Shared_DNN(view2_input)
merged_vector = concatenate([encoded_view1, encoded_view2], axis=-1, name='merged_layer')

model = Model(inputs=[view1_input, view2_input], outputs=merged_vector)
#model.compile(loss=triplet_batch_hard_loss, optimizer=adam_optim)
#model.summary()

for data_xy in cross_fold_generate(config):
    train_audio, train_rgb, train_lab, test_audio, test_rgb, test_lab = data_xy
    print('train_lablllllllllllll',train_lab[:10])
    print(train_audio.shape, len(train_lab),'okokkkkkkkkkkk')
    labels = np.array(list(set(train_lab)))
    model.compile(loss=custom_loss(labels), optimizer=adam_optim)
    model.summary()
    model.fit([train_audio, train_rgb],y=train_lab, validation_data=None, batch_size=32, epochs=10)
    model.save_weights('triplet_MNIST.hdf5')

    trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)
    trained_model.load_weights('triplet_MNIST.hdf5')

    X_test_audio = trained_model.predict(test_audio)
    X_test_rgb   = trained_model.predict(test_rgb)
    views = X_test_audio,X_test_rgb
    corr_matrix = utils.corr_compute(views, tag = "cosine")
    result_list = utils.metric(corr_matrix, test_lab)
    print(result_list)
    mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2, mean_prec_x, mean_rec_x, mean_prec_y, mean_rec_y \
                   = result_list
