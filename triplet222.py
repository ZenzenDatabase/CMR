#!/usr/bin/env python
import numpy as np
import os

import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate, Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam

from itertools import permutations


import h5py
import utils
from myconfig import Myconfig
from keras.utils import to_categorical
margin = 0.5

def pairwise_cosine_sim(a, b):
    similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)
    # Only necessary if vectors are not normalized
    similarity /= tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1)
    # If you prefer the distance measure
    distance = 1 - similarity
    return distance

def batch_hard_triplet_loss(y_true, y_pred, margin=0.5):
    total_lenght = y_pred.shape.as_list()[-1]
    Anchor = y_pred[:,0:int(total_lenght*1/3)]
    Positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    Negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]
    ## hard positive
    anchor_positive_dist  = pairwise_cosine_sim(Anchor, Positive)
    hardest_positive_dist = K.max(anchor_positive_dist, axis=1, keepdims=True)
    ## hard negative
    anchor_negative_dist = pairwise_cosine_sim(Anchor, Negative)
    hardest_negative_dist = K.min(anchor_negative_dist, axis=1, keepdims=True)

    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    triplet_loss = tf.squeeze(triplet_loss)

    #triplet_loss,y = tf.unique(triplet_loss)
    l_left_shift = tf.concat((triplet_loss[1:], [0]), axis=0)
    mask_left_shift = tf.not_equal(triplet_loss - l_left_shift, 0)
    mask = tf.concat(([True], mask_left_shift[:-1]), axis=0)
    triplet_loss = tf.boolean_mask(triplet_loss, mask)

    triplet_loss = K.mean(triplet_loss)
    return triplet_loss
    
def triplet_loss(y_true, y_pred, alpha = 0.5):
    total_lenght = y_pred.shape.as_list()[-1]
    
    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss

def create_base_network(in_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Dense(300, input_dim=in_dims))
    model.add(Activation("tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(300, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(300, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Activation("sigmoid"))
    return model

def gen(audio, rgb, lab):
    for i in range(40):
        yield [audio[:20*(i+1)], rgb[:20*(i+1)], rgb[:20*(i+1)]], lab[:20*(i+1)]

def myGenerator(train_audio, train_rgb, train_lab):
    batch_num = 500
    train_data_xy = [train_audio, train_rgb, train_lab]
    data_idx_lists = range(len(train_audio))
    batchs_data_idxs = np.array_split(data_idx_lists, batch_num)
    print("\nbatch number={}; batch size={}".format(len(batchs_data_idxs), len(batchs_data_idxs[0])))
    i = 0
    for batch_idxs in batchs_data_idxs:
        batch_train_audio = train_audio[batch_idxs]
        batch_train_rgb   = train_rgb[batch_idxs]
        batch_train_lab   = train_lab[batch_idxs]
        
        print("batch-{} ".format(i))
        batch_triplet_list, Lab = [], []
        Anchor_list, Positive_list, Negative_list = [], [], []
        Anchor_x, Positive_x, Negative_x, Lab_x = np.empty((0,10), np.float32), np.empty((0,10), np.float32), np.empty((0,10), np.float32), np.empty((0,), np.float32)
        for data_class in sorted(set(batch_train_lab)):
            ### same idxs
            same_class_idx = np.where(batch_train_lab == data_class)[0]
            ### diff idxs
            diff_class_idx = np.where(batch_train_lab != data_class)[0]
            A_P_pairs = list(permutations(same_class_idx, 2))
            Neg_idx = list(diff_class_idx)
            print("pairs number is {}, neg idx number is {}".format(len(same_class_idx), len(diff_class_idx), len(A_P_pairs), len(Neg_idx)))
            for ap in A_P_pairs:
                Anchor   = np.array(batch_train_audio[ap[0]], dtype=np.float32)
                Positive = batch_train_rgb[ap[1]]
                for Neg_index in Neg_idx:
                    Negative = batch_train_rgb[Neg_index]
                    Anchor_x   = np.concatenate((Anchor_x, [Anchor]))
                    Positive_x = np.concatenate((Positive_x, [Positive]))
                    Negative_x = np.concatenate((Negative_x, [Negative]))
                    Lab_x = np.concatenate((Lab_x, [data_class]))
##                    print(Anchor_x.shape, Positive_x.shape,Negative_x.shape, Lab_x.shape)
        i+=1
        Lab_x = to_categorical(Lab_x)
        print('12345----79798',Anchor_x.shape, Positive_x.shape, Negative_x.shape, Lab_x.shape)
        yield  [Anchor_x, Positive_x, Negative_x], Lab_x #np.array([Anchor_list, Positive_list, Negative_list]), Lab
        
def cross_fold_generate(path):
    files_list = os.listdir(path)
    for i in range(1):# len(files_list)
        new_files_list = files_list[:i]+files_list[i+1:]
        ## read out the file
        ff = h5py.File(path+files_list[i], 'r')
        test_audio, test_rgb = ff["featN1"], ff["featN2"]
        test_lab = ff["testLs"]

        ## train files
        train_audio, train_rgb, train_lab = np.empty((0, 10), np.float32), np.empty((0, 10), np.float32), np.empty((0,), np.float32)
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


if __name__ == "__main__":
    with tf.device('/device:GPU:1'):
        ## Inputs
        anchor_input = Input((10,))
        ##print("anchor input:",anchor_input)
        positive_input = Input((10,))
        ##print("positive input:",positive_input)
        negative_input = Input((10,))
        ##print("negative input:",negative_input)
        ## Neural Networks
        Shared_DNN     = create_base_network(10)
        encoded_anchor = Shared_DNN(anchor_input)
        encoded_positive = Shared_DNN(positive_input)
        encoded_negative = Shared_DNN(negative_input)
        ## Ouputs
        merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
        model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
        ## Compile
        ##print(model.summary())
        sgd = SGD(lr=0.0001, momentum=0.8, decay=0.0, nesterov=False)
        model.compile(loss=triplet_loss, optimizer=sgd)
                 

        config = Myconfig()
        for data_xy in cross_fold_generate(config.path):
            train_audio, train_rgb, train_lab, test_audio, test_rgb, test_lab = data_xy
            metric_lst = []
            ################# model = Models() ########################################
            ## Inputs
            anchor_input = Input((10,))
            print("anchor input:",anchor_input)
            positive_input = Input((10,))
            print("positive input:",positive_input)
            negative_input = Input((10,))
            print("negative input:",negative_input)
            ## Neural Networks
            Shared_DNN = create_base_network(10)
            encoded_anchor = Shared_DNN(anchor_input)
            encoded_positive = Shared_DNN(positive_input)
            encoded_negative = Shared_DNN(negative_input)
            ## Ouputs
            merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
            model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
            ## Compile
            sgd = SGD(lr=0.0001, momentum=0.8, decay=0.0, nesterov=False)
    ##        model.compile(loss=triplet_loss, optimizer=sgd)
            model.compile(loss=batch_hard_triplet_loss, optimizer=sgd)#batch_hard_triplet_loss
    ##        model.fit([train_audio[:1000], train_rgb[:1000],train_rgb[2000:3000]], train_lab[:1000],batch_size=32, epochs=1)
            model.fit_generator(myGenerator(train_audio, train_rgb, train_lab), steps_per_epoch=10, nb_epoch=1)
            model.save_weights('triplet_batchs.hdf5')
            trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)
            trained_model.load_weights('triplet_batchs.hdf5')

            X_test_audio = trained_model.predict(test_audio)
            X_test_rgb   = trained_model.predict(test_rgb)
            print("The output of test:",X_test_audio.shape, X_test_rgb.shape)
            views = X_test_audio, X_test_rgb
            corr_matrix = utils.corr_compute(views, tag = "cosine")
            print("computing the metrics....")
            result_list = utils.metric(corr_matrix, test_lab)
            print(result_list)
            mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2, mean_prec_x, mean_rec_x, mean_prec_y, mean_rec_y \
                           = result_list
            metric_lst.append([mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2])
        print(metric_lst)
        metric_matrix = np.matrix(np.array(metric_lst)).mean(0)  
        result = "(Audio-Visual): map={0}, MRR1={1}\n(Visual-Audio): map={2}, MRR1={3}".\
                     format(metric_matrix.item(0),metric_matrix.item(2), metric_matrix.item(1), metric_matrix.item(3))
        print(result)

