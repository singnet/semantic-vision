import sys
import numpy as np
import cv2
# TODO: replace by configuration parameter
sys.path.insert(0, '/home/vital/projects/vqa/bottom-up-attention/caffe/python')
import caffe
# TODO: replace by configuration parameter
sys.path.insert(0, '/home/vital/projects/vqa/bottom-up-attention/lib')
from fast_rcnn.test import im_detect

from utils import *
from interface import FeatureExtractor

class ImageFeatureExtractor(FeatureExtractor):
    
    def __init__(self, prototxt, weights, imagesPath, imagePrefix):
        self.prototxt = prototxt
        self.weights = weights
        self.imagesPath = imagesPath
        self.imagePrefix = imagePrefix
        self.net = self.initFeatureExtractingNetwork()
    
    def initFeatureExtractingNetwork(self):
        caffe.set_mode_cpu()
        return caffe.Net(self.prototxt, caffe.TEST, weights=self.weights)
    
    def getImageFileName(self, imageId):
        return self.imagePrefix + addLeadingZeros(imageId, 12) + '.jpg'

    def loadImageUsingFileHandle(self, fileHandle):
        data = np.asarray(bytearray(fileHandle.read()), dtype=np.uint8)
        return cv2.imdecode(data)
    
    def loadImageByFileName(self, imageFileName):
        return loadDataFromZipOrFolder(self.imagesPath, imageFileName, 
            lambda fileHandle: self.loadImageUsingFileHandle(fileHandle))
        
    def getFeaturesByImageId(self, imageId):
        image = self.loadImageByFileName(self.getImageFileName(imageId))
        scores, boxes, attr_scores, rel_scores = im_detect(self.net, image)
        pass
