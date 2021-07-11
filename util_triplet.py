import os
import itertools
import numpy as np
from scipy.spatial.distance import euclidean,cosine,cdist

def linear_cca(H1, H2, cat, use_cluster, outdim_size, beta):
    """
    An implementation of linear CCA
    # Arguments:
        H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
        outdim_size: specifies the number of new features
    # Returns
        A and B: the linear transformation matrices 
        mean1 and mean2: the means of data for both views
    """
    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o = H1.shape[1]

    mean1 = np.mean(H1, axis=0)
    mean2 = np.mean(H2, axis=0)
    H1bar = H1 - np.tile(mean1, (m, 1))
    H2bar = H2 - np.tile(mean2, (m, 1))

    SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(H1.shape[1])
    SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(H2.shape[1])
    #print m, o, H1bar.shape, H2bar.shape, SigmaHat11.shape, SigmaHat22.shape
    if (use_cluster):
        FEATDIM = o # 1-demension
        NUMCAT  = 10 # classes
        corr = np.zeros([FEATDIM,FEATDIM], dtype=np.float32)
        num  = 0
        for c in range(NUMCAT):
            flag = (cat==c)
            flags = np.array([int(fg) for fg in flag])
            xi = H1bar[flags,:]     # use centered value
            xj = H2bar[flags,:]
            size = xi.shape
            xiext= np.repeat(xi, [size[0]], axis=0)
            xjext= np.tile(xj, (size[0], 1))
            corr += np.matmul(np.transpose(xiext), xjext)
            num += size[0]*size[0]

        SigmaHat12_c = corr/num
        SigmaHat12_r = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat12 = beta * SigmaHat12_r + (1-beta) * SigmaHat12_c
    else:
        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)


    [D1, V1] = np.linalg.eig(SigmaHat11)
    [D2, V2] = np.linalg.eig(SigmaHat22)
    SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

    Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    [U, D, V] = np.linalg.svd(Tval)
    V = V.T
    A = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
    B = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
    D = D[0:outdim_size]

    return A, B, mean1, mean2
def load_fold_ids(fold_path):
    fold_file = open(fold_path)
    return fold_file.read().splitlines()

def uint8_to_float32(x):
    return (np.float32(x) - 128.0) / 256.0

def read_database(path):
    data = {}
    filenames = os.listdir(path)
    for filename in os.listdir(path):
        features = np.load(path + '/' + filename)
        data[filename[:-4]] = features
    return data

def read_data(path):
    data = {}
    filenames = os.listdir(path)
    sample    = np.load(os.path.join(path, filenames[0]))
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
    return data

def readout_data(audio_path, rgb_path, label_path):
    audio_data, rgb_data, label_data = read_data(audio_path), read_data(rgb_path), read_data(label_path)
    return audio_data, rgb_data, label_data

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
##    leN = len(RelN)
##    prec_col1, rec_col1 = np.empty((0,leN), np.float32), np.empty((0,leN), np.float32)
##    prec_col2, rec_col2 = np.empty((0,leN), np.float32), np.empty((0,leN), np.float32)
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
        imrr_view1, imrr_view2 = imrr_one(pred_view1_label, label), imrr_one(pred_view2_label, label)
        print("AP_view1={}, AP_view2={}; immr_view1={}, immr_view2={}".format(AP_view1, AP_view2,imrr_view1, imrr_view2))
        Ap_sum_view1    += AP_view1
        Ap_sum_view2    += AP_view2
        imrr1_sum_view1 += imrr_view1
        imrr1_sum_view2 += imrr_view2
        
    mAp_view1      = float("{:.5f}".format((Ap_sum_view1*1.0)/query_num))
    mAp_view2      = float("{:.5f}".format((Ap_sum_view2*1.0)/query_num))
    Avg_mrr1_view1 = float("{:.5f}".format((imrr1_sum_view1*1.0)/query_num))
    Avg_mrr1_view2 = float("{:.5f}".format((imrr1_sum_view2*1.0)/query_num))

    print("/nCurrent fold; map_view1={}, map_view2={}, mrr1_view1={}, mrr1_view2={}".format(mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2))
    mean_prec1, mean_rec1 = 0, 0 # np.mean(prec_col1, axis=0), np.mean(rec_col1, axis=0)
    mean_prec2, mean_rec2 = 0, 0 # np.mean(prec_col2, axis=0), np.mean(rec_col2, axis=0)
    return mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2 , mean_prec1, mean_rec1, mean_prec2, mean_rec2

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

