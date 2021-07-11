## metric calculates
import numpy as np
import os, os.path
from scipy.spatial.distance import cdist

RelN = [1, 10, 20, 50, 100, 200, 500, 800, 1200, 1700, 2300,
 2900,
 3500,
 4500,
 5500,
 5620]

def _prec_rec(y_pred, queries_label, label):
    prec_list = []
    rec_list  = []
    for _id in RelN:
        pre_score = 0.0
        count = 0.0
        lab_retrieved = list(queries_label[:_id]).count(label)
        lab_rel = list(queries_label).count(label)
        for i, p in enumerate(y_pred[:_id]):
            if int(p) == int(label):
                count += 1
                pre_score += count/(i+1.0)
        if lab_retrieved != 0:
            prec = (pre_score*1.0)/lab_retrieved
        else:
            prec = 0.0
        if lab_retrieved != 0:
            rec  = (pre_score*1.0)/lab_rel
        else:
            rec  = 0.0
        prec_list.append(prec)
        rec_list.append(rec)
    return prec_list, rec_list

print(RelN)

def corr_compute(views, tag = "cosine"):
    if tag == "cosine":  
        corr_matrix = 1-cdist(views[0], views[1], 'cosine')
    return corr_matrix

def metric(views, queries_label):
    '''
        Input:
            1. corr_matrix (2-D Matrix)
            2. query_labels     (1-D Matrix).
        Output:
            MAP and Avg_MRR1 for view1 as query and view2 as query.
    '''
    corr_matrix = corr_compute(views, tag = "cosine")
    mean_pre_rec_dict = {}
    Ap_sum_view1    = 0.0
    Ap_sum_view2    = 0.0
    imrr1_sum_view1 = 0.0
    imrr1_sum_view2 = 0.0

    query_num = corr_matrix.shape[0]
    leN = len(RelN)
    prec_all1,rec_all1  = [], []
    prec_all2,rec_all2  = [], []
        
    pre_final1, rec_final1 = [], []
    pre_final2, rec_final2 = [], []
    for _idx in range(query_num):
        label = queries_label[_idx]
        rank_view1_index = np.argsort(-corr_matrix[_idx,:])
        rank_view2_index = np.argsort(-corr_matrix[:,_idx])

        pred_view1_label = [queries_label[_index1] for _index1 in rank_view1_index]
        pred_view2_label = [queries_label[_index2] for _index2 in rank_view2_index]

        prec_list1, rec_list1 = _prec_rec(pred_view1_label, queries_label, label)
        prec_list2, rec_list2 = _prec_rec(pred_view2_label, queries_label, label)
        
        AP_view1, AP_view2    = AvgP(pred_view1_label, queries_label, label), AvgP(pred_view2_label, queries_label, label)
        Ap_sum_view1    += AP_view1
        Ap_sum_view2    += AP_view2
        
        prec_all1.append(prec_list1)
        rec_all1.append(rec_list1)
        prec_all2.append(prec_list2)
        rec_all2.append(rec_list2)

    mAp_view1      = float("{:.5f}".format((Ap_sum_view1*1.0)/query_num))
    mAp_view2      = float("{:.5f}".format((Ap_sum_view2*1.0)/query_num))
    print("/nCurrent fold; map_view1={}, map_view2={}".format(mAp_view1, mAp_view2))
    
    prec_all1, rec_all1, prec_all2, rec_all2 = np.array(prec_all1), np.array(rec_all1), np.array(prec_all2), np.array(rec_all2)
    [pre_final1.append(np.mean(prec_all1[:,i])) for i in range(prec_all1.shape[1])]
    [rec_final1.append(np.mean(rec_all1[:,i])) for i in range(rec_all1.shape[1])]
    [pre_final2.append(np.mean(prec_all2[:,i])) for i in range(prec_all2.shape[1])]
    [rec_final2.append(np.mean(rec_all2[:,i])) for i in range(rec_all2.shape[1])]
    return mAp_view1, mAp_view2,pre_final1,rec_final1, pre_final2, rec_final2

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
