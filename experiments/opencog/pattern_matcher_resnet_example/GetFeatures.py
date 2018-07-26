import tensorflow.contrib.keras as ks
from keras.applications.resnet50 import ResNet50
from keras.applications import imagenet_utils
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def getFeatures(imgName):
    feature_extractor = ResNet50(include_top=False)
    img = ks.preprocessing.image.load_img(imgName, target_size=(224,224))
    processedImage = np.asarray([ks.preprocessing.image.img_to_array(img)])
    processedImage = ks.applications.resnet50.preprocess_input(processedImage)
    features = feature_extractor.predict(processedImage)
    features = features.flatten()
    return features

def findWord(inputList, Word):
    idx = -1
    for i in range (1000):
        if inputList[0][i][1] == Word:
            idx = i
    if idx > -1:
        return inputList[0][idx][2]
    else:
        return 0.0

def predicate (features, ID):
    base_model = ResNet50()
    weights = base_model.get_layer(name='fc1000').get_weights()
    result = np.matmul(np.transpose(features), weights[0])
    result = softmax(result)
    result = np.reshape(result, [-1, 1000])
    P = imagenet_utils.decode_predictions(result, 1000)
    probability = findWord(P, ID)
    return probability

def predicateHare (features):
    return predicate(features, 'hare')