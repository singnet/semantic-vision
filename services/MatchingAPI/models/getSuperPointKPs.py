import numpy as np
import argparse
import cv2
import yaml
from pathlib import Path

import experiment
from superpoint.settings import EXPER_PATH

import os
from multiprocessing import Pool
import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
    cv2.circle(img, tuple({100, 500}), radius, color, thickness=-1)
    return img

def select_top_k(prob, thresh=0, num=300):
    pts = np.where(prob > thresh)
    idx = np.argsort(prob[pts])[::-1][:num]
    pts = (pts[0][idx], pts[1][idx])
    return pts

def _getMagicPointKps(image, confidence_threshold):
    nparr = np.fromstring(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="../models/magicpoint_config.yaml", type=str)
    parser.add_argument('--experiment_name', default="magic-point_coco", type=str)

    args = parser.parse_args()

    experiment_name = args.experiment_name

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    assert 'eval_iter' in config

    checkpoint = Path(EXPER_PATH, experiment_name)
    image = np.expand_dims(image, 2)
    with experiment._init_graph(config, with_dataset=False) as (net):
        net.load(str(checkpoint))
        prob = net.predict({'image': image}, keys='prob_nms')
        pts = select_top_k(prob, thresh=confidence_threshold)
        return pts

def getMagicPointKps(image, confidence_threshold):
    with Pool(1) as p:
        return p.apply(_getMagicPointKps, (image, confidence_threshold))

def _getSuperPointKps(image, confidence_threshold):
    nparr = np.fromstring(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="../models/superpoint_config.yaml", type=str)
    parser.add_argument('--experiment_name', default="superpoint_coco", type=str)

    args = parser.parse_args()

    experiment_name = args.experiment_name

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    assert 'eval_iter' in config

    checkpoint = Path(EXPER_PATH, experiment_name)
    image = np.expand_dims(image, 2)
    with experiment._init_graph(config, with_dataset=False) as (net):
        net.load(str(checkpoint))
        prob = net.predict({'image': image}, keys='prob_nms')
        pts = select_top_k(prob, thresh=confidence_threshold)
        return pts

def getSuperPointKps(image, confidence_threshold):
    with Pool(1) as p:
        return p.apply(_getSuperPointKps, (image, confidence_threshold))

def _getSuperPointDescriptors(image, confidence_threshold):
    nparr = np.fromstring(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="../models/superpoint_config.yaml", type=str)
    parser.add_argument('--experiment_name', default="superpoint_coco", type=str)
    args = parser.parse_args()
    experiment_name = args.experiment_name

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    assert 'eval_iter' in config
    checkpoint = Path(EXPER_PATH, experiment_name)
    image = np.expand_dims(image, 2)

    with experiment._init_graph(config, with_dataset=False) as (net):
        net.load(str(checkpoint))

        prob = net.predict({'image': image}, keys=['prob_nms', 'descriptors'])
        pts = select_top_k(prob['prob_nms'], thresh=confidence_threshold)
        desc = prob['descriptors'][pts[0], pts[1]]
        return pts, desc

def getSuperPointDescriptors(image, confidence_threshold):
    with Pool(1) as p:
        return p.apply(_getSuperPointDescriptors, (image, confidence_threshold))
