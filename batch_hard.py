#!/usr/bin/env python
import numpy as np
import os

import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate, Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
import keras

from itertools import permutations
import seaborn as sns

import h5py
import utils
from myconfig import Myconfig
from keras.utils import to_categorical

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.tensorflow_backend._get_available_gpus()

config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

def pairwise_cosine_sim(a, b):
    similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)
    # Only necessary if vectors are not normalized
    similarity /= tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1)
    # If you prefer the distance measure
    distance = 1 - similarity
    return distance

def batch_hard_triplet_loss(y_true, y_pred, margin = 0.1):
    total_lenght = y_pred.shape.as_list()[-1]
    Anchor = y_pred[:,0:int(total_lenght*1/3)]
    Positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    Negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]
    print("Anchor, Positive, Negative:", Anchor, Positive, Negative)
    ## hard positive
    anchor_positive_dist  = pairwise_cosine_sim(Anchor, Positive)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keep_dims=True)
    ## hard negative
    anchor_negative_dist = pairwise_cosine_sim(Anchor, Negative)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keep_dims=True)

    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    triplet_loss = tf.squeeze(triplet_loss, axis=1)

    #triplet_loss,y = tf.unique(triplet_loss)
    l_left_shift = tf.concat((triplet_loss[1:], [0]), axis=0)
    mask_left_shift = tf.not_equal(triplet_loss - l_left_shift, 0)
    mask = tf.concat(([True], mask_left_shift[:-1]), axis=0)
    triplet_loss = tf.boolean_mask(triplet_loss, mask)

    loss = tf.reduce_mean(triplet_loss)
    return loss


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

def gen(audio, rgb, lab):
    for i in range(40):
        yield [audio[:20*(i+1)], rgb[:20*(i+1)], rgb[:20*(i+1)]], lab[:20*(i+1)]

def myGenerator(config, train_audio, train_rgb, train_lab, batch_num):
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
            Anchor_x, Positive_x, Negative_x, Lab_x = np.empty((0, config.input), np.float32), np.empty((0,config.input), np.float32), np.empty((0,config.input), np.float32), np.empty((0,), np.float32)
            for data_class in sorted(set(batch_train_lab)):
                ### same idxs
                same_class_idx = np.where(batch_train_lab == data_class)[0]
                ### diff idxs
                diff_class_idx = np.where(batch_train_lab != data_class)[0]
                A_P_pairs = list(permutations(same_class_idx, 2))
                Neg_idx = list(diff_class_idx)
                #print("same_class sample number is {}, neg idx number is {}, A_P_pairs {}".format(len(same_class_idx), len(diff_class_idx), len(A_P_pairs)))
     
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
        
def cross_fold_generate(config, path):
    file_list = os.listdir(path)

    for i in range(1):
        # train
        train_emb = "train_{0:02d}s".format(i)
        ftr = h5py.File(path+train_emb+".h5", 'r')
        train_audio = ftr["featN1"]
        train_rgb   = ftr["featN2"]
        train_lab = ftr["testLs"]
        train_lab = np.asarray([int(labb) for labb in train_lab])

        # test
        test_emb = "test_{0:02d}s".format(i)
        ff = h5py.File(path+test_emb+".h5", 'r')
        test_audio, test_rgb = ff["featN1"], ff["featN2"]
        test_lab = ff["testLs"]
        print("train, test shape like {},{},{}; {},{},{}".format(train_audio.shape, train_rgb.shape, train_lab.shape, test_audio.shape, test_rgb.shape, test_lab.shape))
        print(type(train_audio), type(train_rgb), type(train_lab), type(test_audio), type(test_rgb), type(test_lab))
        yield np.array(train_audio, dtype=np.float32), np.array(train_rgb, dtype=np.float32), np.array(train_lab, dtype=np.float32), np.array(test_audio, dtype=np.float32), np.array(test_rgb, dtype=np.float32), np.array(test_lab, dtype=np.float32)

##def cross_fold_generate(config, path):
##    files_list = os.listdir(path)
##    for i in range(1):# len(files_list)
##        new_files_list = files_list[:i]+files_list[i+1:]
##        ## read out the file
##        ff = h5py.File(path+files_list[i], 'r')
##        test_audio, test_rgb = ff["featN1"], ff["featN2"]
##        test_lab = ff["testLs"]
##
##        ## train files
##        train_audio, train_rgb, train_lab = np.empty((0, 10), np.float32), np.empty((0, 10), np.float32), np.empty((0,), np.float32)
##        for ft in new_files_list:
##            ftr = h5py.File(path+ft, 'r')
##            ft_audio = ftr["featN1"]
##            ft_rgb   = ftr["featN2"]
##            ft_lab = ftr["testLs"]
##            ft_lab = np.asarray([int(labb) for labb in ft_lab])
##            train_audio = np.concatenate((train_audio, ft_audio))
##            train_rgb   = np.concatenate((train_rgb, ft_rgb))
##            train_lab   = np.concatenate((train_lab, ft_lab))
##            print("train, test shape like {},{},{}; {},{},{}".format(train_audio.shape, train_rgb.shape, train_lab.shape, test_audio.shape, test_rgb.shape, test_lab.shape))
##            print(type(train_audio), type(train_rgb), type(train_lab), type(test_audio), type(test_rgb), type(test_lab))
##        yield np.array(train_audio), np.array(train_rgb), np.array(train_lab), np.array(test_audio), np.array(test_rgb), np.array(test_lab)


if __name__ == "__main__":
    config = Myconfig()   
    print("batch hard .......................................")
    print("batch number: {}".format(config.batch_num))
    i=0
    for data_xy in cross_fold_generate(config, config.path):
        train_audio, train_rgb, train_lab, test_audio, test_rgb, test_lab = data_xy
        print("current fold as test; the train and test infomation:\n",\
              train_audio.shape, train_rgb.shape, train_lab.shape, test_audio.shape, test_rgb.shape, test_lab.shape)
        metric_lst = []

        ## Inputs
        anchor_input = Input((config.input,))
        print("anchor input:",anchor_input)
        positive_input = Input((config.input,))
        print("positive input:",positive_input)
        negative_input = Input((config.input,))
        print("negative input:",negative_input)
        ## Neural Networks
        #Anchor_DNN = create_base_network(config.input)
        Shared_DNN       = create_base_network(config.input)
        encoded_anchor   = Shared_DNN(anchor_input) #Shared_DNN(anchor_input)
        encoded_positive = Shared_DNN(positive_input)
        encoded_negative = Shared_DNN(negative_input)
        ## Ouputs
        merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
        model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
        ## Compile
        adam_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(loss=batch_hard_triplet_loss, optimizer=adam_optim)#batch_hard_triplet_loss
        model.fit_generator(myGenerator(config, train_audio, train_rgb, train_lab, config.batch_num), steps_per_epoch=config.batch_num, nb_epoch=1) #config.batch_num
        
        model.save_weights('triplet_batchs.hdf5')
        trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)
        trained_model.load_weights('triplet_batchs.hdf5')

        X_test_audio = trained_model.predict(test_audio)
        X_test_rgb   = trained_model.predict(test_rgb)
        print("The output of test:",X_test_audio.shape, X_test_rgb.shape)
        views = X_test_audio, X_test_rgb
        corr_matrix = utils.corr_compute(views, tag = "cosine")
        save_path = "batch_hard_cca_correlation_matrix"+str(i)+"_"+str(config.batch_num)
        np.save(save_path, corr_matrix)
        i+=1
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

