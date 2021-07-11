#!/usr/bin/env python
import argparse

import tensorflow
import os, os.path
import numpy as np
import h5py
import utils

argparser = argparse.ArgumentParser(prog='eval', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('-e', '--embed_model_name', default='ccca_vgg_inception') # choices=['ccca_vgg_inception']
argparser.add_argument('-ex', '--embed_model_name_x', default='') 
argparser.add_argument('-t', '--data_embed_tag', choices=["s", "z"], default='s')

args = argparser.parse_args()

print("evaluation on",args.embed_model_name, args.embed_model_name_x)

def save_result(embed_files):
    metric_lst = []
    precision_x, recall_x = np.empty((0, len(utils.RelN)), np.float32), np.empty((0, len(utils.RelN)), np.float32)
    precision_y, recall_y = np.empty((0, len(utils.RelN)), np.float32), np.empty((0, len(utils.RelN)), np.float32)
    for files in embed_files:
        tag = args.data_embed_tag
        ff = h5py.File(files, 'r')
        views = ff["featN1"], ff["featN2"]
        testLb = ff["testLs"]
        print("Current test size = {}".format(len(testLb)))
        corr_matrix = utils.corr_compute(views, tag = "cosine")
        result_list = utils.metric(corr_matrix, testLb)
        mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2 = result_list
        print("MAP for view1={}; MAP for view2={}".format(mAp_view1, mAp_view2))
        metric_lst.append([mAp_view1, mAp_view2, Avg_mrr1_view1, Avg_mrr1_view2])
        
    metric_matrix   = np.matrix(np.array(metric_lst)).mean(0)
    
##    mean_precision_x, mean_recall_x = np.mean(precision_x, axis=0), np.mean(recall_x, axis=0)
##    mean_precision_y, mean_recall_y = np.mean(precision_y, axis=0), np.mean(recall_y, axis=0)
    result = "(Audio-Visual): map={0}, MRR1={1}\n(Visual-Audio): map={2}, MRR1={3}".\
             format(metric_matrix.item(0),metric_matrix.item(2), metric_matrix.item(1), metric_matrix.item(3))
    print(result)
##    precision_recall_result = "(Audio-Visual): precision={0}, recall={1}\n(Visual-Audio): precision={2}, recall={3}".\
##                              format(mean_precision_x, mean_recall_x, mean_precision_y, mean_recall_y)
##    result = result+"\n"+precision_recall_result
    result_filename = files.split("/")[-2]+"_fold"+tag+"_result.npy"
    result_path = os.path.join(utils.RESULT_PATH, result_filename)
    np.save(result_path, result)
        
def run():
    emb_path = os.path.join(utils.LOAD_EMBED_ROOT, args.embed_model_name)
    embed_files = [os.path.join(emb_path, "{0:02d}{1}.h5".format(i, args.data_embed_tag)) for i in range(utils.FOLDS_NUM)]
    print("experiment on:", embed_files)
    save_result(embed_files)

def run_model_fusion():
    emb_path_a = os.path.join(utils.LOAD_EMBED_ROOT, args.embed_model_name)
    embed_file_a = [os.path.join(emb_path_a, "{0:02d}{1}.h5".format(i, args.data_embed_tag)) for i in range(utils.FOLDS_NUM)]
    if args.embed_model_name_x:
        print("Fusion {} and {}".format(args.embed_model_name, args.embed_model_name_x))
        emb_path_b = os.path.join(utils.LOAD_EMBED_ROOT, args.embed_model_name_x)
        embed_file_b = [os.path.join(emb_path_b, "{0:02d}{1}.h5".format(i, args.data_embed_tag)) for i in range(utils.FOLDS_NUM)]
        print(embed_file_a, embed_file_b)
        saving(embed_file_a, embed_file_b)
    else:
        save_result(embed_file_a)

def saving(embed_file_a, embed_file_b):
    metric_lst = []
    for file_a, file_b in zip(embed_file_a, embed_file_b):
        tag = args.data_embed_tag
        fa = h5py.File(file_a, 'r')
        view_a = fa["featN1"], fa["featN2"]
        testLa = fa["testLs"]

        fb = h5py.File(file_b, 'r')
        view_b = fb["featN1"], fb["featN2"]
        testLb = fb["testLs"]
        print("Current test size = {};{}".format(len(testLa), len(testLb)))

        views = utils.fusion_lst(view_a[0], view_b[0], normalization="one_zero"), utils.fusion_lst(view_a[1], view_b[1], normalization="one_zero")
        corr_matrix = utils.corr_compute(views, tag = "cosine")
        print(utils.metric(corr_matrix, testLb))
        metric_lst.append(utils.metric(corr_matrix, testLb))
    metric_matrix = np.matrix(np.array(metric_lst)).mean(0)
    result = "(Audio-Visual): map={0}, MRR1={1}\n(Visual-Audio): map={2}, MRR1={3}".\
             format(metric_matrix.item(0),metric_matrix.item(2), metric_matrix.item(1), metric_matrix.item(3))
    print(result)
    
    result_filename = file_a.split("/")[-2]+"_fold"+tag+"_result.npy"
    result_path = os.path.join(utils.RESULT_PATH, result_filename)
    np.save(result_path, result)

if __name__=="__main__":
    run_model_fusion()
