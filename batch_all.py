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
#import seaborn as sns

import h5py
import util_triplet as utl
from myconfig import Myconfig
from keras.utils import to_categorical
import itertools

margin = 0.5

print("batch all ............................")
def pairwise_cosine_sim(a, b):
    similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)
    # Only necessary if vectors are not normalized
    similarity /= tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1)
    # If you prefer the distance measure
    distance = 1 - similarity
    return distance

def batch_hard_triplet_loss(y_true, y_pred, margin=0.3):
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

def batch_all_triplet_loss(y_true, y_pred, margin=0.3):
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
    triplet_loss = tf.cast(triplet_loss, tf.float32)
    triplet_loss = triplet_loss/(num_positive_triplets+1e-16)
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

def create_pn_network(in_dims):
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
    model.add(Dropout(0.))
    model.add(Activation("sigmoid"))
    return model

def create_an_network(in_dims):
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

def myGenerator(train_audio, train_rgb, train_lab, batch_num):
##    batch_num = 500
    train_data_xy = [train_audio, train_rgb, train_lab]
    data_idx_lists = range(len(train_audio))
    batchs_data_idxs = np.array_split(data_idx_lists, batch_num)
    print("\nbatch number={}; batch size={}".format(len(batchs_data_idxs), len(batchs_data_idxs[0])))
    i = 0
    while True:
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
                #print("pairs number is {}, neg idx number is {}".format(len(same_class_idx), len(diff_class_idx), len(A_P_pairs), len(Neg_idx)))
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
            print('Batch valid triplets shape:',Anchor_x.shape, Positive_x.shape, Negative_x.shape, Lab_x.shape)
            yield  [Anchor_x, Positive_x, Negative_x], Lab_x #np.array([Anchor_list, Positive_list, Negative_list]), Lab

def linearccca_emb(train_audio_rgb_label, test_audio_rgb_label, output_size, beta):
    audioTrain, rgbTrain, labelTrain = train_audio_rgb_label
    audioTest,  rgbTest,  labelTest  = test_audio_rgb_label
    trainDccaModel = True
    if (trainDccaModel):
        print("Linear CCA started!")
        w0, w1, m0, m1 = utl.linear_cca(audioTrain, rgbTrain, labelTrain, False, output_size, beta)
        np.savez('Model_Cc_CCA.npz', w0=w0, w1=w1, m0=m0, m1=m1)
        print("Linear CCA ended!")
    else:
        model = np.load('Model_Cc_CCA.npz')
        w0 = model['w0']
        w1 = model['w1']
        m0 = model['m0']
        m1 = model['m1']

    data_num = len(audioTrain)
    audioTrain -= m0.reshape([1, -1]).repeat(data_num, axis=0)
    audioFeatTrain = np.dot(audioTrain, w0)
    rgbTrain -= m1.reshape([1, -1]).repeat(data_num, axis=0)
    rgbFeatTrain = np.dot(rgbTrain, w1)
    print("Embedding the train data!", audioFeatTrain.shape, rgbFeatTrain.shape, labelTrain.shape)

    data_num = len(audioTest)
    audioTest -= m0.reshape([1, -1]).repeat(data_num, axis=0)
    audioFeatTest = np.dot(audioTest, w0)
    rgbTest -= m1.reshape([1, -1]).repeat(data_num, axis=0)
    rgbFeatTest = np.dot(rgbTest, w1)
    print("Embedding the train data!", audioFeatTest.shape, rgbFeatTest.shape, labelTest.shape)
    return audioFeatTrain, rgbFeatTrain, labelTrain, audioFeatTest, rgbFeatTest, labelTest
        
def cross_fold_embed_read():
    audio_path = '/home/dhzeng/AVIDEO/Data/vgg/'
    rgb_path = '/home/dhzeng/AVIDEO/Data/inception/'
    label_path = '/home/dhzeng/AVIDEO/Data/VEGAS_classes/'

    fold_path = '/home/dhzeng/AVIDEO/Data/folds/'

    audio_data, rgb_data, label_data = utl.readout_data(audio_path, rgb_path, label_path)
    


    for i in range(1):
        folds_lst = os.listdir(fold_path)
        train_vid = list(itertools.chain(*[utl.load_fold_ids(os.path.join(fold_path, folds_lst[j])) for j in range(5) if j != i]))
        test_vid = utl.load_fold_ids(os.path.join(fold_path, folds_lst[i]))

        (train_audio, test_audio), (train_rgb, test_rgb), (train_label, test_label) = utl.train_test_split(audio_data, \
                     train_vid, test_vid), utl.train_test_split(rgb_data, train_vid, test_vid), \
                                           utl.train_test_split(label_data, train_vid, test_vid)
        train_audio_rgb_label = train_audio, train_rgb, train_label
        test_audio_rgb_label  = test_audio, test_rgb, test_label
        output_size = 10
        beta        = 0.2
        audioFeatTrain, rgbFeatTrain, labelTrain, audioFeatTest, rgbFeatTest, labelTest = linearccca_emb(train_audio_rgb_label, test_audio_rgb_label, output_size,\
                                                             beta)
        yield audioFeatTrain, rgbFeatTrain, labelTrain, audioFeatTest, rgbFeatTest, labelTest
       # yield np.array(train_audio, dtype=np.float32), np.array(train_rgb, dtype=np.float32), np.array(train_label,
        #                                                                                               dtype=np.float32), np.array(
         #   test_audio, dtype=np.float32), np.array(test_rgb, dtype=np.float32), np.array(test_label, dtype=np.float32)


if __name__ == "__main__":
    ## Inputs
    anchor_input = Input((10,))
    ##print("anchor input:",anchor_input)
    positive_input = Input((10,))
    ##print("positive input:",positive_input)
    negative_input = Input((10,))
    ##print("negative input:",negative_input)
    ## Neural Networks
    Anchor_DNN     = create_an_network(10)
    Shared_DNN     = create_pn_network(10)
    encoded_anchor = Anchor_DNN(anchor_input)
    encoded_positive = Shared_DNN(positive_input)
    encoded_negative = Shared_DNN(negative_input)
    ## Ouputs
    merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
    Anchor_branch = Model(inputs=anchor_input, outputs=encoded_anchor)
    Pos_neg_branch = Model(inputs=positive_input, outputs=encoded_positive)
    ## Compile
    ##print(model.summary())
    adam_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss=triplet_loss, optimizer=adam_optim)
             

    config = Myconfig()
    print("batch all .......................................")
    print("batch number: {}".format(config.batch_num))
    i=0
    for data_xy in cross_fold_embed_read():
        train_audio, train_rgb, train_lab, test_audio, test_rgb, test_lab = data_xy
        viewx = test_audio, test_rgb
        c_matrix = utl.corr_compute(viewx, tag = "cosine")
        result_l = utl.metric(c_matrix, test_lab)
        print(result_l)
        print("current fold as test; the train and test infomation:\n",\
              train_audio.shape, train_rgb.shape, train_lab.shape, test_audio.shape, test_rgb.shape, test_lab.shape)
        metric_lst = []
        model.fit_generator(myGenerator(train_audio, train_rgb, train_lab, config.batch_num), steps_per_epoch=config.batch_num, nb_epoch=10)
        
        Anchor_branch.save_weights("encoded_anchor.hdf5")
        Pos_neg_branch.save_weights("encoded_pos_neg.hdf5")

        trained_anchor = Model(inputs=anchor_input, outputs=encoded_anchor)
        trained_pos_neg = Model(inputs=positive_input, outputs=encoded_positive)

        trained_anchor.load_weights('encoded_anchor.hdf5')
        trained_pos_neg.load_weights('encoded_pos_neg.hdf5')

        X_test_audio = trained_anchor.predict(test_audio)
        X_test_rgb   = trained_pos_neg.predict(test_rgb)
        print("The output of test:",X_test_audio.shape, X_test_rgb.shape)
        views = X_test_audio, X_test_rgb
        corr_matrix = utl.corr_compute(views, tag = "cosine")
        i+=1
        print("computing the metrics....")
        result_list = utl.metric(corr_matrix, test_lab)
        print(result_list)
        mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2, mean_prec_x, mean_rec_x, mean_prec_y, mean_rec_y \
                       = result_list
        metric_lst.append([mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2])
    print(metric_lst)
    metric_matrix = np.matrix(np.array(metric_lst)).mean(0)  
    result = "(Audio-Visual): map={0}, MRR1={1}\n(Visual-Audio): map={2}, MRR1={3}".\
                 format(metric_matrix.item(0),metric_matrix.item(2), metric_matrix.item(1), metric_matrix.item(3))
    print(result)

