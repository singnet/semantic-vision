
from __future__ import print_function

import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import _pickle as cPickle
import numpy as np
import math
import pandas as pd


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infile = 'data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'

FEATURESNAMES = ['roi_x', 'roi_y' 'roi_width' 'roi_height' 'spatial_feature_6d' 'img_feature_d2048']

train_indices_file = 'data/train36_imgid2idx.pkl'
val_indices_file = 'data/val36_imgid2idx.pkl'
train_ids_file = 'data/train_ids.pkl'
val_ids_file = 'data/val_ids.pkl'

train_parsed_path = 'data/train2014_parsed_features'
val_parsed_path = 'data/val2014_parsed_features'
test_parsed_path = 'data/test2015_parsed_features'


feature_length = 2048
num_fixed_boxes = 36
num_spatial_features = 6

# FEATURESNAMES = ['roi_x', 'roi_y' 'roi_width' 'roi_height' 'spatial_feature_6d' 'img_feature_d2048']
def load_parsed_features(pathFeatures, imgIDSet, filePrefix = 'COCO_train2014_', id_len = 12, reduce_set=False):
    data=[]

    nImg = len(imgIDSet)

    # !! FOR DEBUG LOAD ONLY 1% OF DATA
    if(reduce_set is True):
        nImg = int(nImg / 100)

    # Read parsed files and accumulate data
    for i in range(nImg):
        image_id = imgIDSet[i]
        filePath = pathFeatures + '/' + filePrefix
        nZeros = int((id_len - 1) - math.floor(math.log10(image_id)))
        for _ in range(0, nZeros):
            filePath = filePath + '0'

        filePath = filePath + str(image_id) + '.tsv'

        df = np.genfromtxt(filePath, delimiter='\t', skip_header=1)
        data.append([image_id, df])

        sys.stdout.write("\r \r Loading"
                         " parsed features: {0}%\ti = {1}/{2}".format( (str(int(100 * float(i) / float(nImg)))), i, nImg ))
        sys.stdout.flush()
        time.sleep(0.01)

    print("\nFeatures loading is completed!")


    return data


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids



