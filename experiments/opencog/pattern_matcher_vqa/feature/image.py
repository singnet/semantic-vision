import sys
import numpy as np
import cv2
import caffe
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg

from util import *
from interface import FeatureExtractor

class ImageFeatureExtractor(FeatureExtractor):
    
    def __init__(self, prototxt, weights, imagesPath, imagePrefix):
        self.prototxt = prototxt
        self.weights = weights
        self.imagesPath = imagesPath
        self.imagePrefix = imagePrefix
        self.net = self.initFeatureExtractingNetwork()
        self.conf_thresh = 0.2
        self.MIN_BOXES = 36
        self.MAX_BOXES = 36

    
    def initFeatureExtractingNetwork(self):
        caffe.set_mode_cpu()
        return caffe.Net(self.prototxt, caffe.TEST, weights=self.weights)
    
    def getImageFileName(self, imageId):
        return self.imagePrefix + addLeadingZeros(imageId, 12) + '.jpg'

    def loadImageUsingFileHandle(self, fileHandle):
        data = np.asarray(bytearray(fileHandle.read()), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    def loadImageByFileName(self, imageFileName):
        return loadDataFromZipOrFolder(self.imagesPath, imageFileName, 
            lambda fileHandle: self.loadImageUsingFileHandle(fileHandle))
        
    def getFeaturesByImageId(self, imageId):
        return self.getFeaturesByImagePath(self.getImageFileName(imageId))
    
    def getFeaturesByImagePath(self, imagePath):
        image = self.loadImageByFileName(imagePath)
        scores, boxes, attr_scores, rel_scores = im_detect(self.net, image)
        
        # Keep the original boxes, don't worry about the regresssion bbox outputs
        rois = self.net.blobs['rois'].data.copy()
        # unscale back to raw image space
        _, im_scales = _get_blobs(image, None)
    
        cls_boxes = rois[:, 1:5] / im_scales[0]
        cls_prob = self.net.blobs['cls_prob'].data
        pool5 = self.net.blobs['pool5_flat'].data
    
        # Keep only the best detections
        max_conf = np.zeros((rois.shape[0]))
        for cls_ind in range(1,cls_prob.shape[1]):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    
        keep_boxes = np.where(max_conf >= self.conf_thresh)[0]
        if len(keep_boxes) < self.MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:self.MIN_BOXES]
        elif len(keep_boxes) > self.MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:self.MAX_BOXES]
        
        features = []
        for box in pool5[keep_boxes]:
            features.append(box.tolist())
        
        return features
