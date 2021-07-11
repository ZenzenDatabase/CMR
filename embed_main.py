#!/usr/bin/env python
import argparse
import utils

import os.path
import subprocess

import numpy as np
import random
import numpy.random
import tensorflow as tf
import keras

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#random.seed(2018)
#np.random.seed(2018)
#tf.set_random_seed(2018)
#os.environ['PYTHONHASHSEED'] = '2018'

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

print(utils.PROJECT_ROOT)
parser = argparse.ArgumentParser(prog='embed_main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m',  '--model', choices=['cca', "dcca", 'ccca', "sdcca", 'seg_lstm_cca','s_dcca'], default='ccca')
parser.add_argument('-a',  '--audio_path',        default='vgg')
parser.add_argument('-af', '--audio_fusion_path', default='') # audio_feat/audio_soundnet.
parser.add_argument('-v',  '--rgb_path',          default='inception') # visual_feat/visual_i3d_joint.
parser.add_argument('-vf', '--rgb_fusion_path',   default='')
parser.add_argument('-o',  '--output_size',   type=int)
parser.add_argument('-b',  '--beta',          type=float)
parser.add_argument('-p',  '--padding',  action='store_true', default=False)
parser.add_argument('-f',  '--fold_path',         default='folds')

args = parser.parse_args()

ROOT = "/home/dhzeng/AVIDEO/Data"
## read out dataset......
audio_path = os.path.join(ROOT, args.audio_path)
if args.audio_fusion_path:
    audio_fusion_path = os.path.join(ROOT, args.audio_fusion_path)
else:
    audio_fusion_path = ""
rgb_path = os.path.join(ROOT, args.rgb_path)
if args.rgb_fusion_path:
    rgb_fusion_path = os.path.join(ROOT, args.rgb_fusion_path)
else:
    rgb_fusion_path = ""
fold_path = os.path.join(ROOT, args.fold_path)
print(args.model)
print(args.padding)
if args.model == "seg_lstm_cca":
    utils.run(audio_path, audio_fusion_path, rgb_path, rgb_fusion_path, fold_path, args.model, args.output_size, args.beta, 2, args.padding)
elif args.padding==True:
    utils.run(audio_path, audio_fusion_path, rgb_path, rgb_fusion_path, fold_path, args.model, args.output_size, args.beta, 1, args.padding)
elif args.padding==False:
    utils.run(audio_path, audio_fusion_path, rgb_path, rgb_fusion_path, fold_path, args.model, args.output_size, args.beta, 2, args.padding)
