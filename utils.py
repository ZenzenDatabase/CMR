import os
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
##import model
from sklearn import preprocessing
import subprocess
from collections import OrderedDict
import itertools
import tensorflow as tf
from myconfig import Myconfig

config = Myconfig()
PROJECT_ROOT    = config.root
FEATURE_ROOT    = config.feat_root
FOLD_ROOT       = config.fold_root
LABEL_PATH      = config.lab_root
LOAD_EMBED_ROOT = config.loademb_root
RESULT_PATH     = config.result_root
FOLDS_NUM       = config.fold_num
TIMES = 1000
RelN =  [5, 50, 100, 150, 200, 250, 300, 400, 500,  600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]  # ranked list

def mv(origin_path, target_path):
    line = "mv "+os.path.join(origin_path, "*")+" "+target_path
    subprocess.call(line, shell=True)

def cp(origin_path, target_path):
    line = "cp "+os.path.join(origin_path, "*")+" "+target_path
    subprocess.call(line, shell=True)

def fusion(data1, data2, normalization="one_zero"):
    fold_path = os.path.join(FOLD_ROOT, "folds/")
    folds_lst = os.listdir(fold_path)
    print([os.path.join(fold_path, folds_lst[i])for i in range(5)])
    fold_vids  = np.array([load_fold_ids(os.path.join(fold_path, folds_lst[i])) for i in range(5)])
    fold_vid  = []
    for i in range(5):
        fold_vid.extend(fold_vids[i])
    order_data1 = OrderedDict((key, data1[key]) for key in data1 if key in fold_vid)
    order_data2 = OrderedDict((key, data2[key]) for key in data2) # if key in fold_vid)
    print(len(order_data1), len(order_data2))
    key1 = list(order_data1.keys())
    new_data2 = OrderedDict([(key, order_data2[key]) for key in order_data1])
    new_key2 = list(new_data2.keys())
    assert key1==new_key2, "same order!"
    value1 = order_data1.values()
    value2 = new_data2.values()
    if normalization=="one_zero":
        print(len(list(order_data1.values())))
        nor_value1 = preprocessing.scale(list(order_data1.values()))
        nor_value2 = preprocessing.scale(list(new_data2.values()))
    elif normalization==None:
        nor_value1 = order_data1.values()
        nor_value2 = new_data2.values()

    new_value = np.concatenate((nor_value1, nor_value2), axis=1)
    data = dict(zip(key1, new_value))
    return data

def fusion_lst(lista, listb, normalization="one_zero"):
    if normalization=="one_zero":
        nor_value1 = preprocessing.scale(lista)
        nor_value2 = preprocessing.scale(listb)
    new_list = np.concatenate((nor_value1, nor_value2), axis=1)
    return new_list

def normalization(x_train, name="one_unit"):
    if name == "one_unit":
        x_normalized = preprocessing.scale(x_train)
    if name == "l2":
        x_normalized = preprocessing.normalize(x_train, norm='l2')
    mean = x_normalized.mean(axis=0)
    std  = x_normalized.std(axis=0)
    return x_normalized

def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)
    if pad_size <= 0:
        return array[:10]
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)
    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)
    return b


def read_data(path, output_dimension=1):
    data = {}
    filenames = os.listdir(path)
    sample    = np.load(os.path.join(path, filenames[0]))
    if output_dimension==1:
        if len(sample.shape) == 2:
            ## read out as one-dimension mapped and normalizated
            for filename in os.listdir(path):
                features = uint8_to_float32(np.mean(np.load(path + '/' + filename), axis=0))
                data[filename[:-4]] = features
        else:
            ## read out as it is
            for filename in os.listdir(path):
                features = np.load(path + '/' + filename)
                data[filename[:-4]] = features
    elif output_dimension==2:
        ## padding with 10*128/10*1024
        assert len(sample.shape) == 2, "The input data should be two-dimension!"
        for filename in os.listdir(path):
            features = np.float32(np.load(path + '/' + filename))
            data[filename[:-4]] = pad_along_axis(features, 10, axis=0)        
    return data

def read_database(path):
    data = {}
    filenames = os.listdir(path)
    for filename in os.listdir(path):
        features = np.load(path + '/' + filename)
        data[filename[:-4]] = features
    return data
    
def readout_data(audio_path, rgb_path, label_path, output_dimension, padding):
    '''
    Input:
        audio_path/rgb_path---The data from this path can be two-dimension or one-dimension.
        label_path------------The label can be a single value or a vector.
    Output:
        One-dimension with float32.
    Function:
        If the input data is two-dimension and output_dimension=1, do mean as one-dimension.
    '''
    if padding:
        audio_data, rgb_data, label_data = read_data(audio_path, output_dimension), read_data(rgb_path, output_dimension), read_data(label_path, 1)
    else:
        audio_data, rgb_data, label_data = read_data(audio_path, 1), read_database(rgb_path), read_data(label_path, 1)
    return audio_data, rgb_data, label_data

def uint8_to_float32(x):
    return (np.float32(x) - 128.0) / 256.0

def load_fold_ids(fold_path):
    fold_file = open(fold_path)
    return fold_file.read().splitlines()

def train_test_split(data_dict, train_vid, test_vid):
    if not list(data_dict.values())[0].shape:
        ## label
        Train, Test = np.empty((0), np.float32), np.empty((0), np.float32)
    elif len(list(data_dict.values())[0].shape)==2:
        ## when input data dimmension = 2
        data_len =  list(data_dict.values())[0].shape
        Train, Test = np.empty((0, data_len[0], data_len[1]), np.float32), np.empty((0, data_len[0], data_len[1]), np.float32)
    elif len(list(data_dict.values())[0].shape)==1:
        ## when input data dimmension = 1
        data_len =  list(data_dict.values())[0].shape[0]
        Train, Test = np.empty((0,data_len), np.float32), np.empty((0,data_len), np.float32)
    for vid in data_dict:
        data_item = np.expand_dims(data_dict[vid], axis=0)
        if vid in train_vid:
            Train = np.concatenate((Train, data_item), axis=0)
        if vid in test_vid:
             Test = np.concatenate((Test, data_item), axis=0)
    print("Trainning set and Testing shape:", Train.shape, Test.shape)
    return Train.astype('float32'), Test.astype('float32')

def extend_train_set(train_audio_rgb_label, times):
    train_audio, train_rgb, train_label = train_audio_rgb_label
    labels = np.asarray([int(ca) for ca in train_label])
    for lab in range(10):
        index_list = np.where(labels==lab)

        #audio
        audio_sample = train_audio[index_list,:][0]
        re_audio_sample = np.repeat(audio_sample, times, axis=0)
        new_train_audio = np.concatenate((train_audio, re_audio_sample))

        #rgb
        rgb_seed_sample = np.empty((0, 1024), np.float32)
        for i in range(len(index_list[0])):
            indexs = np.delete(index_list[0], [i])[:times]
            try:
                rgb_seed_sample = np.concatenate((rgb_seed_sample, train_rgb[indexs,:]))
            except:
                print(rgb_seed_sample.shape, train_rgb[indexs,:].shape)
        new_train_rgb   = np.concatenate((train_rgb, rgb_seed_sample))
        new_train_label = np.concatenate((train_label, [train_label[lab]]*(times*audio_sample.shape[0])))
    return new_train_audio, new_train_rgb, new_train_label

def train_test_split_com(audio_data, rgb_data, label_data, train_vid, test_vid):
    train_label, test_label = np.empty((0), np.float32), np.empty((0), np.float32)
    audio_data_len =  list(audio_data.values())[0].shape[0]
    rgb_data_len   =  list(rgb_data.values())[0].shape[1] 
    train_audio, test_audio = np.empty((0,audio_data_len), np.float32), \
                              np.empty((0,audio_data_len), np.float32)
    train_rgb, test_rgb     = np.empty((0,rgb_data_len), np.float32), \
                              np.empty((0,rgb_data_len), np.float32)
    
    if 'baby_cry_video_00850' in audio_data.keys():
        print("you are the best")
    for vid in rgb_data:
        
        if vid in train_vid:
            seg_len = rgb_data[vid].shape[0]
            audio_data_item = np.expand_dims(audio_data[vid], axis=0)
            label_data_item = np.expand_dims(label_data[vid], axis=0)
            train_rgb   = np.concatenate((train_rgb, rgb_data[vid]), axis=0)
            for i in range(int(seg_len)):
                train_audio = np.concatenate((train_audio, audio_data_item), axis=0)
                train_label = np.concatenate((train_label, label_data_item), axis=0)
        if vid in test_vid:
            features = uint8_to_float32(np.mean(rgb_data[vid], axis=0))
            audio_data_item = np.expand_dims(audio_data[vid], axis=0)
            rgb_data_item   = np.expand_dims(features, axis=0)
            label_data_item = np.expand_dims(label_data[vid], axis=0)
            test_rgb   = np.concatenate((test_rgb, [features]), axis=0)
            test_audio = np.concatenate((test_audio, audio_data_item), axis=0)
            test_label = np.concatenate((test_label, label_data_item), axis=0)
    return (train_audio.astype('float32'), test_audio.astype('float32')), (train_rgb.astype('float32'), test_rgb.astype('float32')), (train_label, test_label)

def save_embedding(loadfile, data, lab):
    testLb = np.array([np.argmax(te) for te in lab])
    hf=h5py.File(loadfile,"w")
    hf.create_dataset("featN1", data=data[0])
    hf.create_dataset("featN2", data=data[1])
    hf.create_dataset("testLs", data=lab) #testLs
    hf.close()

def corr_compute(views, tag = "cosine"):
    if tag == "cosine":  
        corr_matrix = 1-cdist(views[0], views[1], 'cosine')
    return corr_matrix

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def cosine_similarity_tf(x, y):
    s = tf.losses.cosine_distance(tf.nn.l2_normalize(x, 0), tf.nn.l2_normalize(y, 0), dim=0)
    return tf.Session().run(s)

## metric calculates
def metric(corr_matrix, queries_label):
    '''
        Input:
            1. corr_matrix (2-D Matrix)
            2. query_labels     (1-D Matrix).
        Output:
            MAP and Avg_MRR1 for view1 as query and view2 as query.
    '''
    
    mean_pre_rec_dict = {}
    Ap_sum_view1    = 0.0
    Ap_sum_view2    = 0.0
    imrr1_sum_view1 = 0.0
    imrr1_sum_view2 = 0.0

    query_num = corr_matrix.shape[0]
    leN = len(RelN)
    prec_col1, rec_col1 = np.empty((0,leN), np.float32), np.empty((0,leN), np.float32)
    prec_col2, rec_col2 = np.empty((0,leN), np.float32), np.empty((0,leN), np.float32)
    for _idx in range(query_num):
        label = queries_label[_idx]
        rank_view1_index = np.argsort(-corr_matrix[_idx,:])
        rank_view2_index = np.argsort(-corr_matrix[:,_idx])

        pred_view1_label = [queries_label[_index1] for _index1 in rank_view1_index]
        pred_view2_label = [queries_label[_index2] for _index2 in rank_view2_index]
        
##        prec_narr1, rec_narr1 = _prec_rec(RelN, pred_view1_label, queries_label, label)
##        prec_narr2, rec_narr2 = _prec_rec(RelN, pred_view2_label, queries_label, label)
##        prec_col1, rec_col1   = np.concatenate((prec_col1, [prec_narr1]), axis=0), \
##                                np.concatenate((prec_col1, [rec_narr1]), axis=0)
##        prec_col2, rec_col2   = np.concatenate((prec_col2, [prec_narr2]), axis=0), \
##                                np.concatenate((prec_col2, [rec_narr2]), axis=0)
        
        AP_view1, AP_view2     = AvgP(pred_view1_label, queries_label, label), AvgP(pred_view2_label, queries_label, label)
#        imrr_view1, imrr_view2 = imrr_one(pred_view1_label, label), imrr_one(pred_view2_label, label)
        #print("AP_view1={}, AP_view2={}; immr_view1={}, immr_view2={}".format(AP_view1, AP_view2,imrr_view1, imrr_view2))
        Ap_sum_view1    += AP_view1
        Ap_sum_view2    += AP_view2
#        imrr1_sum_view1 += imrr_view1
#        imrr1_sum_view2 += imrr_view2
        
    mAp_view1      = float("{:.5f}".format((Ap_sum_view1*1.0)/query_num))
    mAp_view2      = float("{:.5f}".format((Ap_sum_view2*1.0)/query_num))
#    Avg_mrr1_view1 = float("{:.5f}".format((imrr1_sum_view1*1.0)/query_num))
#    Avg_mrr1_view2 = float("{:.5f}".format((imrr1_sum_view2*1.0)/query_num))


#    mean_prec1, mean_rec1 = 0, 0 # np.mean(prec_col1, axis=0), np.mean(rec_col1, axis=0)
#    mean_prec2, mean_rec2 = 0, 0 # np.mean(prec_col2, axis=0), np.mean(rec_col2, axis=0)
    return mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2#, mean_prec1, mean_rec1, mean_prec2, mean_rec2

def _prec_rec(RelN, y_pred, queries_label, label):
    prec_list = []
    rec_list  = []
    for _id in RelN:
        score = 0.0
        count = 0.0
        lab_retrieved = list(queries_label[:_id]).count(label)
        lab_rel = list(queries_label).count(label)
##        print(_id, lab_retrieved, lab_rel)
        for i, p in enumerate(y_pred[:_id]):
            if int(p) == int(label):
                count += 1
                score += count/(i+1.0)
        if lab_retrieved != 0:
            prec = (score*1.0)/lab_retrieved
        else:
            prec = 0.0
        if lab_retrieved != 0:
            rec  = (score*1.0)/lab_rel
        else:
            rec  = 0.0
        prec_list.append(prec)
        rec_list.append(rec)
    return np.asarray(prec_list), np.asarray(rec_list)

def imrr_one(y_pred, label):
    '''
        Input:
            1. prediction  (1-D matrix).
            2. label   (A label (int)).
        Output:
            A single value, the mean of the mrr1 of all queries, for each retrieval.
    '''
    return 1.0 / (np.argwhere(np.array(y_pred) == label)[0,0] + 1) # imrr1

def AvgP(y_pred, queries_label, label):
    '''
        Input:
            1. prediction (1-D matrix).
            2. label   (A label (int)).
        Output:
            A single value, the mean of the mrr1 of all queries.
    '''
    score = 0.0
    count = 0.0

    lab_rel = list(queries_label).count(label)
    for i, p in enumerate(y_pred):
        if int(p) == int(label):
            count += 1
            score += count/(i+1.0)
    if lab_rel == 0:
        avgP = 0.0
    else:
        avgP = (score*1.0)/lab_rel
    return avgP

def read_dataset(path, tag="feature"):
    data = {}
    filenames = os.listdir(path)
    if tag=="feature":
        for filename in os.listdir(path):
            features = uint8_to_float32(np.mean(np.load(path + '/' + filename), axis=0))
            data[filename[:-4]] = features
    elif tag=="label" or None:
        for filename in os.listdir(path):
            features = np.load(path + '/' + filename)
            data[filename[:-4]] = features
    return data

def load_fold_ids(fold_path):
    fold_file = open(fold_path)
    return fold_file.read().splitlines()

def train_test_spliting(data_dict, train_vid, test_vid):
    Train, Test = [], []    
    for vid in data_dict:
        if vid in train_vid:
            Train.append(data_dict[vid])
        if vid in test_vid:
            Test.append(data_dict[vid])
    return np.array(Train, dtype='float32'), np.array(Test, dtype='float32')

ROOT = "/home/dhzeng/AVIDEO/Data"
fold_path = os.path.join(ROOT, "folds")
folds_lst = os.listdir(fold_path)
    
audio_path = os.path.join(ROOT, "vgg")
rgb_path   = os.path.join(ROOT, "inception")
lab_path   = os.path.join(ROOT, "VEGAS_classes")
    
outdim_size = 10


def run(audio_path, audio_fusion_path, rgb_path, rgb_fusion_path, fold_path, model_name, output_size, beta, output_dimension, padding):
    audio_data, rgb_data, label_data = read_dataset(audio_path, tag="feature"), read_dataset(rgb_path, tag="feature"), read_dataset(lab_path, tag="label")
    lst = filter(None, [model_name, audio_path.split("/")[-1][6:], \
        audio_fusion_path.split("/")[-1][6:],rgb_path.split("/")[-1][7:], rgb_fusion_path.split("/")[-1][7:]])
    emb_item = "_".join(lst)
    if model_name!="cca":
        LOAD_EMBED_ROOT_X = str(output_size)+"_"+str(beta).replace(".", "")
    else:
        LOAD_EMBED_ROOT_X=''
    load_emb_path = os.path.join(LOAD_EMBED_ROOT, emb_item)+LOAD_EMBED_ROOT_X
    if not os.path.exists(load_emb_path):
        os.makedirs(load_emb_path)
#    audio_data, rgb_data, label_data = readout_data(audio_path, rgb_path, LABEL_PATH, output_dimension, padding)

##    print('audio_data; rgb_data; dimemsion as follow.', audio_data.values()[100].shape, rgb_data.values()[100].shape, label_data.values()[0].shape)
#    if audio_fusion_path:
#        audio_fusion_data = read_data(audio_fusion_path, output_dimension, padding)
#        audio_data = fusion(audio_data, audio_fusion_data, normalization="one_zero")
#    if rgb_fusion_path:
#        rgb_fusion_data = read_data(rgb_fusion_path, output_dimension, padding)
#        rgb_data = fusion(rgb_data, rgb_fusion_data, normalization="one_zero")
    folds_lst = os.listdir(fold_path)
    testModel = False
    print("Start to training")
    for i in range(FOLDS_NUM):
        train_vid = list(itertools.chain(*[load_fold_ids(os.path.join(fold_path, folds_lst[j])) for j in range(5) if j!=i]))
        test_vid = load_fold_ids(os.path.join(fold_path, folds_lst[i]))
        print("train, test size:", len(train_vid), len(test_vid))
        assert list(set(train_vid).intersection(set(test_vid))) == [], "You train data and test data exist overlap! please check!"
        
        if not padding:
            print("Start to split...............")
            (train_audio, test_audio), (train_rgb, test_rgb), (train_label, test_label) \
                          = train_test_spliting(audio_data, train_vid, test_vid),\
                            train_test_spliting(rgb_data,   train_vid, test_vid),\
                            train_test_spliting(label_data, train_vid, test_vid)
#        else:
#            (train_audio, test_audio), (train_rgb, test_rgb), (train_label, test_label) = train_test_split_com(audio_data, rgb_data, label_data, train_vid, test_vid)
        
        times = TIMES
#        if times:
#            train_audio_rgb_label = extend_train_set((train_audio, train_rgb, train_label), times)
#            print('After extend.....',train_audio_rgb_label[0].shape, train_audio_rgb_label[1].shape)
#        else:
#            train_audio_rgb_label = train_audio, train_rgb, train_label
        train_audio_rgb_label = train_audio, train_rgb, train_label
        test_audio_rgb_label  = test_audio,  test_rgb,  test_label
        print("Train data size:{},{},{}; Test data size:{}, {}, {}".format(train_audio.shape,  train_rgb.shape, len(train_label), \
                                                                           test_audio.shape, test_rgb.shape, len(test_label)))
        if model_name == "cca":
            featN, testLs = model.cca_emb(train_audio_rgb_label, test_audio_rgb_label, load_emb_path)
        if model_name == "dcca":
            featN, testLs = model.dcca_emb(train_audio_rgb_label, test_audio_rgb_label, load_emb_path, output_size, beta)
        if model_name == "ccca":
            featN, testLs = model.ccca_emb(train_audio_rgb_label, test_audio_rgb_label, testModel, load_emb_path, output_size, beta)
        if model_name == "sdcca":
            featN, testLs = model.sdcca_emb(train_audio_rgb_label, test_audio_rgb_label, load_emb_path, output_size, beta)
        if model_name == "seg_lstm_cca":
            featN, testLs = model.seg_lstm_cca_emb(train_audio_rgb_label, test_audio_rgb_label, load_emb_path, output_size, beta)
        if model_name == "s_dcca":
            featN, testLs = model.sdcca_embedding(train_audio_rgb_label, test_audio_rgb_label, load_emb_path, output_size, beta)
            
        emb = "{0:02d}".format(i)
        tag = fold_path.split("/")[-1][-1]
        loadfile = os.path.join(load_emb_path, emb+tag+".h5")
        save_embedding(loadfile, [featN[0], featN[1]], testLs)
