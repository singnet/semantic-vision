import logging
import torch
import torch.nn.functional as F

from utils import *

class NetsVocabularyNeuralNetworkRunner():
    
    def __init__(self, modelsFileName):
        self.logger = logging.getLogger('NetsVocabularyNeuralNetworkRunner')
        self.netsVocabulary = self.loadNets(modelsFileName)

    def loadNets(self, modelsFileName):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(modelsFileName, map_location=device.type)
        netsVocabulary = netsvocabularyModule.NetsVocab.fromStateDict(device, checkpoint['state_dict'])
        netsVocabulary.train(False)
        return netsVocabulary
    
    def runNeuralNetwork(self, features, word):
        model = self.netsVocabulary.getModelByWord(word)
        if model is None:
            self.logger.debug('no model found, return FALSE')
            return torch.zeros(1)
        # TODO: F.sigmoid should part of NN
        return F.sigmoid(model(torch.Tensor(features)))

# import netsvocabulary
netsvocabularyModule = importModuleFromFile('netsvocabulary',
          currentDir(__file__) + '/../DNNs/vqa_multi_dnn/netsvocabulary.py')
