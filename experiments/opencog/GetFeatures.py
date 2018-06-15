import tensorflow.contrib.keras as ks
from keras.applications.resnet50 import ResNet50
import numpy as np

def getFeatures(imgName):

    feature_extractor = ResNet50(include_top=False)
    img = ks.preprocessing.image.load_img(imgName, target_size=(224,224))
    processedImage = np.asarray([ks.preprocessing.image.img_to_array(img)])
    processedImage = ks.applications.resnet50.preprocess_input(processedImage)
    features = feature_extractor.predict(processedImage)
    features = features.flatten()
    return features